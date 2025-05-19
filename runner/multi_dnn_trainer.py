import torch
from tqdm import tqdm

from base import BaseTrainer
from .tracker import MetricTracker


class Multi_DNN_Trainer(BaseTrainer):
    '''
    Multi DNN Trainer
    '''
    def __init__(self, config, train_data_loader, valid_data_loader=None, test_data_loader=None):
        '''
        Initialize the Multi DNN Trainer instance with parameters.

        Needed parameters
        :param config: The configuration dictionary.
        '''
        super().__init__(config)

        # Set models and optimizers for easy access.
        for model_name in self.models:
            if model_name == 'Feature_Extractor' or model_name == 'Genomic_Separate_Feature_Extractor':
                extractor = self.models[model_name]
                opt_extractor = self.optimizers[model_name]
            elif model_name == 'Label_Classifier' or model_name == 'Task_Classifier':
                classifier = self.models[model_name]
                opt_classifier = self.optimizers[model_name]
            else:
                raise KeyError(f'Unknown model name: {model_name}')
        self.list_models = (extractor, classifier)
        self.list_optimizers = (opt_extractor, opt_classifier)

        # Dataloaders
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        # Trainer parameters settings
        self.len_epoch = len(self.train_data_loader)
        self.bootstrap_project_id_indices = config['runner'].get('bootstrap_project_id_indices', [''])

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
            iter_keys=[m.__name__ if i == '' else f'{i}_{m.__name__}'
                       for m in self.metrics for i in self.bootstrap_project_id_indices],
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
            'project_id': [],
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
                for bootstrap_project_id_index in self.bootstrap_project_id_indices:
                    if metric.__name__ == 'c_index':
                        if bootstrap_project_id_index == '':
                            self.test_metrics.iter_update(metric.__name__,
                                                          metric(outputs, survival_times, vital_statuses))
                        else:
                            self.test_metrics.iter_update(
                                f'{bootstrap_project_id_index}_{metric.__name__}',
                                metric(
                                    outputs[project_ids == bootstrap_project_id_index],
                                    survival_times[project_ids == bootstrap_project_id_index],
                                    vital_statuses[project_ids == bootstrap_project_id_index]
                                )
                            )
                    else:
                        if bootstrap_project_id_index == '':
                            self.test_metrics.iter_update(metric.__name__, metric(outputs, targets))
                        else:
                            self.test_metrics.iter_update(
                                f'{bootstrap_project_id_index}_{metric.__name__}',
                                metric(
                                    outputs[project_ids == bootstrap_project_id_index],
                                    targets[project_ids == bootstrap_project_id_index]
                                )
                            )
            # Record bootstrap status
            if isinstance(genomics, torch.Tensor):
                if 'genomic' not in bootstrap_status:
                    bootstrap_status['genomic'] = []
                bootstrap_status['genomic'].append(genomics.numpy())
            bootstrap_status['clinical'].append(clinicals.numpy())
            bootstrap_status['index'].append(idx.numpy())
            bootstrap_status['project_id'].append(project_ids.numpy())
            bootstrap_status['output'].append(outputs.numpy())
            bootstrap_status['target'].append(targets.numpy())
            bootstrap_status['survival_time'].append(survival_times.numpy())
            bootstrap_status['vital_status'].append(vital_statuses.numpy())

        # Save bootstrap statuses
        self._save_bootstrap_status(self.test_data_loader.dataset.project_ids, bootstrap_status)

        return self.test_metrics.result()
