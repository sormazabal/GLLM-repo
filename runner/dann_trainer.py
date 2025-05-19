import torch
from base import BaseTrainer
from .tracker import MetricTracker


class DANN_Trainer(BaseTrainer):
    '''
    DANN Trainer
    '''
    def __init__(self, config, train_data_loader, valid_data_loader=None, test_data_loader=None):
        '''
        Initialize the DANN Trainer instance with parameters.

        Needed parameters
        :param config: The configuration dictionary.
        '''
        super().__init__(config)

        # Dataloaders
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        # Trainer parameters settings
        self.len_epoch = len(self.train_data_loader)
        self.bootstrap_project_id_index = config['runner'].get('bootstrap_project_id_index', None)

        # Metric trackers
        self.train_metrics = MetricTracker(
            epoch_keys=[m.__name__ for m in self.metrics],
            iter_keys=['loss'], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            epoch_keys=[m.__name__ for m in self.metrics],
            iter_keys=[], writer=self.writer
        )
        self.test_metrics = MetricTracker(
            iter_keys=[m.__name__ for m in self.metrics],
            epoch_keys=[], writer=self.writer
        )

    def _train_epoch(self, epoch):
        '''
        Training logic for an epoch

        :param epoch: Current epoch number
        '''
        # Set models to training mode
        for model_name in self.models:
            self.models[model_name].train()

        # Reset train metric tracker
        self.train_metrics.reset()

        # Start training
        label_outputs = []
        domain_outputs = []
        targets = []
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            genomic, clinical, project_id = data

            # Transfer device and dtype
            genomic = genomic.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
            clinical = clinical.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
            project_id = project_id.to(self.device, dtype=torch.int64, non_blocking=self.non_blocking)
            target = target.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)

            # Normalize per iteration
            genomic = (genomic - genomic.mean(dim=0)) / genomic.std(dim=0)
            genomic = torch.nan_to_num(genomic, 0)

            # Extract Features
            if self.models.get('Feature_Extractor', None) is not None:
                feature_extractor_name = 'Feature_Extractor'
                embeddings = self.models['Feature_Extractor'](genomic, clinical)
            elif self.models.get('Genomic_Seperate_Feature_Extractor', None) is not None:
                feature_extractor_name = 'Genomic_Seperate_Feature_Extractor'
                embeddings = self.models['Genomic_Seperate_Feature_Extractor'](genomic, clinical, project_id)
            else:
                raise KeyError('Please specify a valid feature extractor')

            # Train Domain Classifier
            self.optimizers['Domain_Classifier'].zero_grad()
            domain_output = self.models['Domain_Classifier'](embeddings.detach())
            domain_loss = self.losses['cross_entropy'](domain_output, project_id)
            domain_loss.backward()
            self.optimizers['Domain_Classifier'].step()

            # Train Feature Extractor and Label Classifier
            label_output = self.models['Label_Classifier'](embeddings)
            domain_output = self.models['Domain_Classifier'](embeddings)
            self.optimizers[feature_extractor_name].zero_grad()
            self.optimizers['Label_Classifier'].zero_grad()
            label_loss = self.losses['bce_with_logits_loss'](label_output, target)
            domain_loss = self.losses['cross_entropy'](domain_output, project_id)
            loss = label_loss - 0.05 * domain_loss
            loss.backward()
            self.optimizers[feature_extractor_name].step()
            self.optimizers['Label_Classifier'].step()

            # Append output and target
            label_outputs.append(label_output.detach().cpu())
            domain_outputs.append(domain_output.detach().cpu())
            targets.append(target.detach().cpu())

            # Update train metric tracker for one iteration
            self.train_metrics.iter_update('loss', loss.item())

        # Set step and record time in tensorboard.
        self.writer.set_step(epoch, mode='train')

        # Concatenate the outputs and targets
        label_outputs = torch.cat(label_outputs)
        domain_outputs = torch.cat(domain_outputs)
        targets = torch.cat(targets)

        # Update train metric tracker for one epoch
        for metric in self.metrics:
            self.train_metrics.epoch_update(metric.__name__, metric(label_outputs, targets))
        self.train_metrics.epoch_update('loss')

        # Update log for one epoch
        log = {'train_'+k: v for k, v in self.train_metrics.result().items()}

        # Validation
        if self.valid_data_loader:
            valid_log = self._valid_epoch(epoch)
            log.update(**{'valid_'+k: v for k, v in valid_log.items()})

        # Update learning rate if there is lr scheduler
        if self.lr_schedulers is not None:
            for lr_scheduler_name in self.lr_schedulers:
                self.lr_schedulers[lr_scheduler_name].step()

        # Testing
        if epoch == self.epochs:
            if self.test_data_loader:
                test_log = self._bootstrap()
                log.update(**{'bootstrap_'+k: v for k, v in test_log.items()})

        return log

    def _valid_epoch(self, epoch):
        '''
        Validating after training an epoch

        :param epoch: Current epoch number
        '''
        # Set models to validating mode
        for model_name in self.models:
            self.models[model_name].eval()

        # Reset valid metric tracker
        self.valid_metrics.reset()

        # Start validating
        label_outputs = []
        domain_outputs = []
        project_ids = []
        targets = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                genomic, clinical, project_id = data

                # Transfer device and dtype
                genomic = genomic.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                clinical = clinical.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                project_id = project_id.to(self.device, dtype=torch.int64, non_blocking=self.non_blocking)
                target = target.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)

                # Normalize per batch
                genomic = (genomic - genomic.mean(dim=0)) / genomic.std(dim=0)
                genomic = torch.nan_to_num(genomic, 0)

                # Extract Features
                if self.models.get('Feature_Extractor', None) is not None:
                    embeddings = self.models['Feature_Extractor'](genomic, clinical)
                elif self.models.get('Genomic_Seperate_Feature_Extractor', None) is not None:
                    embeddings = self.models['Genomic_Seperate_Feature_Extractor'](genomic, clinical, project_id)
                else:
                    raise KeyError('Please specify a valid feature extractor')

                # Label Classifier
                label_output = self.models['Label_Classifier'](embeddings)

                # Domain Classifier
                domain_output = self.models['Domain_Classifier'](embeddings)

                # Append output and target
                label_outputs.append(label_output.detach().cpu())
                domain_outputs.append(domain_output.detach().cpu())
                project_ids.append(project_id.detach().cpu())
                targets.append(target.detach().cpu())

            # Set step and record time in tensorboard.
            self.writer.set_step(epoch, mode='valid')

            # Concatenate the outputs and targets
            label_outputs = torch.cat(label_outputs)
            domain_outputs = torch.cat(domain_outputs)
            project_ids = torch.cat(project_ids)
            targets = torch.cat(targets)

            # Update valid metric tracker for one epoch
            for metric in self.metrics:
                self.valid_metrics.epoch_update(metric.__name__, metric(label_outputs, targets))

        return self.valid_metrics.result()

    def _bootstrap(self, repeat_times=1000):
        '''
        Testing logic for a model with bootstrapping.

        :param model_name: Current model name.
        '''
        # Set models to validating mode
        for model_name in self.models:
            self.models[model_name].eval()

        # Reset test metric tracker
        self.test_metrics.reset()

        # Start validating
        for idx in range(repeat_times):
            label_outputs = []
            project_ids = []
            targets = []
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_data_loader):
                    genomic, clinical, project_id = data

                    # Transfer device and dtype
                    genomic = genomic.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                    clinical = clinical.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                    project_id = project_id.to(self.device, dtype=torch.int64, non_blocking=self.non_blocking)
                    target = target.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)

                    # Normalize per batch
                    genomic = (genomic - genomic.mean(dim=0)) / genomic.std(dim=0)
                    genomic = torch.nan_to_num(genomic, 0)

                    # Extract Features
                    if self.models.get('Feature_Extractor', None) is not None:
                        embeddings = self.models['Feature_Extractor'](genomic, clinical)
                    elif self.models.get('Genomic_Seperate_Feature_Extractor', None) is not None:
                        embeddings = self.models['Genomic_Seperate_Feature_Extractor'](genomic, clinical, project_id)
                    else:
                        raise KeyError('Please specify a valid feature extractor')

                    # Label Classifier
                    label_output = self.models['Label_Classifier'](embeddings)

                    # Append output and target
                    label_outputs.append(label_output.detach().cpu())
                    project_ids.append(project_id.detach().cpu())
                    targets.append(target.detach().cpu())

                # Concatenate the outputs and targets
                label_outputs = torch.cat(label_outputs)
                project_ids = torch.cat(project_ids)
                targets = torch.cat(targets)

                # Update valid metric tracker for one epoch
                for metric in self.metrics:
                    if self.bootstrap_project_id_index is not None:
                        self.test_metrics.iter_update(
                            metric.__name__,
                            metric(
                                label_outputs[project_ids == self.bootstrap_project_id_index],
                                targets[project_ids == self.bootstrap_project_id_index]
                            )
                        )
                    else:
                        self.test_metrics.iter_update(metric.__name__, metric(label_outputs, targets))

        return self.test_metrics.result()
