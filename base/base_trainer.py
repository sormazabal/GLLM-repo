from __future__ import annotations
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader

import model as module_model
import utils.runner.loss as module_loss
import utils.runner.metric as module_metric
import utils.runner.plot as module_plot
from base import BaseModel
from parse_config import ConfigParser
from utils.logger import TensorboardWriter, get_logger


class BaseTrainer(object):
    '''
    Base class for all trainers
    '''
    def __init__(self, config: ConfigParser):
        self.config = config
        self.logger = get_logger('runner.base_trainer')
        self.train_data_loader: DataLoader
        self.valid_data_loader: DataLoader
        self.test_data_loader: DataLoader
        self.list_models: list[type[BaseModel]]
        self.list_optimizers: list[torch.optim.Optimizer]

        # Setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.non_blocking = config['pin_memory']

        # Setup models.
        # The `name` and `type` should be reversed, but `config.init_obj` thinks otherwise. Blame the legacy code.
        self.models: dict[str, type[BaseModel]] = {
            model_name: config.init_obj(
                f'models.{model_name}',
                module_model
            ).to(self.device) for model_name in config['models']
        }

        if len(device_ids) > 1:
            for model_name in self.models:
                self.models[model_name] = torch.nn.parallel.DataParallel(self.models[model_name], device_ids=device_ids)

        # Setup optimizers
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        for optimizer_name in self.config['optimizers']:
            # There should be one and only one model for each optimizer.
            # Don't know why the legacy code allows multiple models.
            params = self.models[optimizer_name].parameters()
            self.optimizers[optimizer_name] = self.config.init_obj(f'optimizers.{optimizer_name}', torch.optim, params)

        # Setup learning rate schedulers
        if self.config['lr_schedulers'] is not None:
            self.lr_schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
            for lr_scheduler_name in self.config['lr_schedulers']:
                optimizer_name = self.config[f'lr_schedulers.{lr_scheduler_name}.optimizer']
                self.lr_schedulers[lr_scheduler_name] = self.config.init_obj(
                    f'lr_schedulers.{lr_scheduler_name}',
                    torch.optim.lr_scheduler,
                    self.optimizers[optimizer_name]
                )
        else:
            self.lr_schedulers = None

        # Loss functions
        self.losses = {loss: config.init_obj(f'losses.{loss}', module_loss) for loss in config['losses']}

        # Metric functions
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        # Plot functions
        self.plots = [getattr(module_plot, plt) for plt in config['plots']]

        # Trainer parameters settings
        self.epochs = config['runner.epochs']
        self.save_epoch = config['runner'].get('save_epoch', self.epochs)
        self.log_epoch = config['runner'].get('log_epoch', 1)
        self.start_epoch = 1
        self.checkpoint_dir = config.ckpt_dir

        # Configuration to monitor model performance and save best
        self.monitor = config['runner'].get('monitor', 'off')
        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = np.inf if self.monitor_mode == 'min' else -np.inf
            self.early_stop = config['runner'].get('early_stop', np.inf)

        # Setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, config['runner.tensorboard'])

        # Resume from the checkpoint
        if config.resume is not None:
            self._load_checkpoint(config.resume)

        # Load from pre-trained model. This is different from resume.
        # FIXME: This is a temporary solution, but the legacy code is too messy to fix.
        try:
            self._load_checkpoint(**config['pretrained'])
        except KeyError:
            pass

    def _train_epoch(self, epoch: int):
        # Set models to training mode
        for model_name in self.models:
            self.models[model_name].train()

        # Reset train metric tracker
        self.train_metrics.reset()

        # Start training
        _, _, _, _, outputs, targets, survival_times, vital_statuses = self._send_data(
            mode='train',
            epoch=epoch,
            dataloader=self.train_data_loader,
        )

        # Update train metric tracker for one epoch
        for metric in self.metrics:
            if metric.__name__ == 'c_index':
                self.train_metrics.epoch_update(metric.__name__, metric(outputs, survival_times, vital_statuses))
            else:
                self.train_metrics.epoch_update(metric.__name__, metric(outputs, targets))
        self.train_metrics.epoch_update('loss')

        # Update log for one epoch
        log = {'train_' + k: v for k, v in self.train_metrics.result().items()}

        # Validation
        if self.valid_data_loader:
            valid_log = self._valid_epoch(epoch)
            log.update(**{'valid_' + k: v for k, v in valid_log.items()})

        # Update learning rate if there is lr scheduler
        if self.lr_schedulers is not None:
            for lr_scheduler_name in self.lr_schedulers:
                self.lr_schedulers[lr_scheduler_name].step()
        return log

    @torch.no_grad()
    def _valid_epoch(self, epoch: int):
        # Set models to validating mode
        for model_name in self.models:
            self.models[model_name].eval()

        # Reset valid metric tracker
        self.valid_metrics.reset()

        # Start validating
        _, _, _, _, outputs, targets, survival_times, vital_statuses = self._send_data(
            mode='valid',
            epoch=epoch,
            dataloader=self.valid_data_loader,
        )

        # Update valid metric tracker for one epoch
        for metric in self.metrics:
            if metric.__name__ == 'c_index':
                self.valid_metrics.epoch_update(metric.__name__, metric(outputs, survival_times, vital_statuses))
            else:
                self.valid_metrics.epoch_update(metric.__name__, metric(outputs, targets))
        self.valid_metrics.epoch_update('loss')

        return self.valid_metrics.result()

    @abstractmethod
    def _bootstrap(self, repeat_times=1000):
        raise NotImplementedError

    def run(self):
        '''
        Full training logic
        '''
        results = {}

        # FIXME: This is a temporary solution to check whether the first model is trained or not.
        init_params: dict[str, np.ndarray] = {}
        params: torch.Tensor
        for name, params in self.list_models[0].named_parameters():
            init_params[name] = params.detach().cpu().numpy()

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Run training for one epoch
            result = self._train_epoch(epoch)

            # Save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # Evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off':
                try:
                    monitor_metric_mean = log[self.monitor_metric]['mean']
                    # Check whether model performance improved or not, according to specified metric(monitor_metric)
                    improved = (self.monitor_mode == 'min' and monitor_metric_mean <= self.monitor_best) or \
                               (self.monitor_mode == 'max' and monitor_metric_mean >= self.monitor_best)
                except KeyError:
                    self.logger.warning(f'Metric {self.monitor_metric} is not found.')
                    self.logger.warning('Model performance monitoring is disabled.')
                    self.monitor_mode = 'off'
                    improved = False

                if improved:
                    self.monitor_best = monitor_metric_mean
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(f'Validation performance didn\'t improve for {self.early_stop} epochs.')
                    self.logger.info('Training stops.')
                    break

            # Update results
            if self.monitor_mode == 'off':
                results.update(result)
            else:
                if best:
                    results.update(result)

            # Log informations
            if epoch % self.log_epoch == 0 or best:
                for key, value in log.items():
                    if isinstance(value, dict):
                        self.logger.info('{:20s}: {:.5f} ±{:.5f}'.format(str(key).lower(), value['mean'], value['std']))
                    else:
                        self.logger.info('{:20s}: {}'.format(str(key).lower(), value))

            # Save model
            if epoch % self.save_epoch == 0 or best:
                self._save_checkpoint(epoch, save_best=best)

        # FIXME: This is a temporary solution to check whether the first model is trained or not.
        for name, params in self.list_models[0].named_parameters():
            final_params = params.detach().cpu().numpy()
            if 'bias' not in name and np.allclose(init_params[name], final_params):
                self.logger.info(f'{name} initial: {init_params[name]}')
                self.logger.info(f'{name} final: {final_params}')

        # Bootstrap
        if self.test_data_loader:
            test_log = self._bootstrap()
            results.update(**{f'bootstrap_{k}': v for k, v in test_log.items()})

        return results

    def _send_data(self, mode: str, epoch: int, dataloader: DataLoader):
        genomics = []
        clinicals = []
        indices = []
        project_ids = []
        outputs = []
        targets = []
        survival_times = []
        vital_statuses = []
        for batch_idx, (data, target) in enumerate(dataloader):
            genomic, clinical, index, project_id = data
            target, survival_time, vital_status = target

            # Transfer device and dtype.
            genomic = genomic.to(self.device, non_blocking=self.non_blocking)
            clinical = clinical.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
            project_id = project_id.to(self.device, dtype=torch.int64, non_blocking=self.non_blocking)
            target = target.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
            survival_time = survival_time.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
            vital_status = vital_status.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)

            # Extract Features
            embedding = self.list_models[0](genomic, clinical, project_id)

            # Label Classifier
            output = self.list_models[1](embedding, project_id)

            # Loss for one batch.
            if mode == 'train':
                self.list_optimizers[0].zero_grad()
                self.list_optimizers[1].zero_grad()
                loss = self.losses['bce_with_logits_loss'](output, target)
                loss.backward()
                self.list_optimizers[0].step()
                self.list_optimizers[1].step()
            else:
                loss = self.losses['bce_with_logits_loss'](output, target)

            # Append output and target
            if isinstance(genomic, torch.Tensor):
                genomics.append(genomic.detach().cpu())
            clinicals.append(clinical.detach().cpu())
            indices.append(index.detach().cpu())
            project_ids.append(project_id.detach().cpu())
            outputs.append(output.detach().cpu())
            targets.append(target.detach().cpu())
            survival_times.append(survival_time.detach().cpu())
            vital_statuses.append(vital_status.detach().cpu())

            # Update train/valid metric tracker for one iteration
            if mode == 'train':
                self.train_metrics.iter_update('loss', loss.item())
            elif mode == 'valid':
                self.valid_metrics.iter_update('loss', loss.item())

        self.writer.set_step(epoch, mode=mode)

        # Concatenate the outputs and targets
        if isinstance(genomic, torch.Tensor):
            genomics = torch.cat(genomics)
        clinicals = torch.cat(clinicals)
        indices = torch.cat(indices)
        project_ids = torch.cat(project_ids)
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        survival_times = torch.cat(survival_times)
        vital_statuses = torch.cat(vital_statuses)
        return genomics, clinicals, indices, project_ids, outputs, targets, survival_times, vital_statuses

    def _prepare_device(self, n_gpu_use: int):
        '''
        Setup GPU device if available, move model into configured device
        '''
        # Get the GPU counts
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning('There\'s no GPU available on this machine, training will be performed on CPU.')
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(f'The number of GPU is configured to be {n_gpu_use}, but only {n_gpu} are available.')
            n_gpu_use = n_gpu

        # Get the device
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch: int, save_best=False):
        '''
        Save checkpoint

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        '''
        state = {
            'epoch': epoch,
            'models': {model_name: model.state_dict() for model_name, model in self.models.items()},
            'optimizers': {optimizer_name: opt.state_dict() for optimizer_name, opt in self.optimizers.items()},
            'monitor_best': self.monitor_best,
            # 'config': self.config
        }

        if self.lr_schedulers is not None:
            state.update({'lr_schedulers': {
                scheduler_name: scheduler.state_dict() for scheduler_name, scheduler in self.lr_schedulers.items()
            }})

        normal_path = self.checkpoint_dir.joinpath(f'checkpoint-epoch{epoch}.pth')
        torch.save(state, normal_path)
        self.logger.info('Saving checkpoint: {}...'.format(normal_path))

        if save_best:
            best_path = self.checkpoint_dir.joinpath('checkpoint-best.pth')
            torch.save(state, best_path)
            self.logger.info('Saving current best checkpoint: {}...'.format(best_path))

    def _load_checkpoint(self, load_path: str, resume=True, model_names: list[str] | str = 'ALL'):
        '''
        Load checkpoint

        :param load_path: Checkpoint path to be loaded
        :param resume: Decide if this checkpoint used for resume the training
        '''
        self.logger.info('Loading checkpoint: {}...'.format(load_path))

        checkpoint = torch.load(load_path)
        self.logger.info('Checkpoint loaded')

        if model_names == 'ALL':
            for model_name in self.models:
                self.models[model_name].load_state_dict(checkpoint['models'][model_name])
        else:
            for model_name in model_names:
                self.models[model_name].load_state_dict(checkpoint['models'][model_name])
        self.logger.info('Loading {} models'.format(model_names))

        if resume:
            self.start_epoch = checkpoint['epoch'] + 1
            self.monitor_best = checkpoint['monitor_best']

            for optimizer_name in self.optimizers:
                self.optimizers[optimizer_name].load_state_dict(checkpoint['optimizers'][optimizer_name])

            if self.lr_schedulers is not None:
                for scheduler_name in self.lr_schedulers:
                    self.lr_schedulers[scheduler_name].load_state_dict(checkpoint['lr_schedulers'][scheduler_name])

            self.logger.info('Resume training from epoch {}'.format(self.start_epoch))

    def _save_bootstrap_status(self, project_ids: list[str], bootstrap_status: dict[str, torch.Tensor]):
        '''
        Save bootstrap statuses

        :param bootstrap_statuses:
        '''
        for k, v in bootstrap_status.items():
            bootstrap_status[k] = np.stack(v, axis=0)

        project_ids = '_'.join(project_ids)
        project_ids = project_ids.lower()

        np.savez(self.config.log_dir.joinpath(f'{project_ids}_bootstrap_status.npz'), **bootstrap_status)
        self.logger.info('Saving bootstrap status to {}'.format(self.config.log_dir))
