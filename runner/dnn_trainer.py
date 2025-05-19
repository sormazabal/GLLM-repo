import torch
from tqdm import tqdm

from base import BaseTrainer
from .tracker import MetricTracker


class DNN_Trainer(BaseTrainer):
    '''
    DNN Trainer
    '''
    def __init__(self, config, train_data_loader, valid_data_loader=None, test_data_loader=None):
        '''
        Initialize the DNN Trainer instance with parameters.

        Needed parameters
        :param config: The configuration dictionary.
        '''
        super().__init__(config)

        # Setup models and optimizers. Only certain models are allowed. Others should use `Multi_DNN_Trainer`.
        self.list_models = (self.models['Feature_Extractor'], self.models['Label_Classifier'])
        self.list_optimizers = (self.optimizers['Feature_Extractor'], self.optimizers['Label_Classifier'])

        # Dataloaders
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        # Trainer parameters settings
        self.len_epoch = len(self.train_data_loader)

        # Metric trackers
        self.train_metrics = MetricTracker(
            epoch_keys=[m.__name__ for m in self.metrics],
            iter_keys=['loss'], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            epoch_keys=[m.__name__ for m in self.metrics],
            iter_keys=['loss'], writer=self.writer
        )
        self.test_metrics = MetricTracker(
            iter_keys=[m.__name__ for m in self.metrics],
            epoch_keys=[], writer=self.writer
        )

    @torch.no_grad()
    def _bootstrap(self, repeat_times=1000):
        '''
        Testing logic for a model with bootstrapping.

        :param repeat_times: Repeated times.
        '''
        # Set models to validating mode
        for model_name in self.models:
            self.models[model_name].eval()

        # Reset test metric tracker
        self.test_metrics.reset()

        # Start bootstrapping
        bootstrap_status = {
            'clinical': [],
            'index': [],
            'output': [],
            'target': [],
            'survival_time': [],
            'vital_status': []
        }
        for _ in tqdm(range(repeat_times), postfix='Bootstrap'):
            genomics, clinicals, idx, project_ids, outputs, targets, survival_times, vital_statuses = self._send_data(
                mode='test',
                epoch=0,
                dataloader=self.test_data_loader,
            )

            # Update test metric tracker for one epoch
            for metric in self.metrics:
                if metric.__name__ == 'c_index':
                    self.test_metrics.iter_update(metric.__name__, metric(outputs, survival_times, vital_statuses))
                else:
                    self.test_metrics.iter_update(metric.__name__, metric(outputs, targets))

            # Record bootstrap status
            if isinstance(genomics, torch.Tensor):
                if 'genomic' not in bootstrap_status:
                    bootstrap_status['genomic'] = []
                bootstrap_status['genomic'].append(genomics.numpy())
            bootstrap_status['clinical'].append(clinicals.numpy())
            bootstrap_status['index'].append(idx.numpy())
            bootstrap_status['output'].append(outputs.numpy())
            bootstrap_status['target'].append(targets.numpy())
            bootstrap_status['survival_time'].append(survival_times.numpy())
            bootstrap_status['vital_status'].append(vital_statuses.numpy())

        # Save bootstrap statuses
        self._save_bootstrap_status([self.test_data_loader.dataset.project_id], bootstrap_status)

        # Save bootstrap models
        self._save_checkpoint(f'_{self.test_data_loader.dataset.project_id.lower()}')

        return self.test_metrics.result()
