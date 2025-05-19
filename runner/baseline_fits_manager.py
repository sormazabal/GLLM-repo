import torch
from base.base_fits_manager import BaseFitsManager
from .tracker import MetricTracker


class Baseline_Fits_Manager(BaseFitsManager):
    '''
    Baseline fits managers
    '''
    def __init__(self, config, train_data_loader, valid_data_loader=None, test_data_loader=None):
        '''
        Initialize the Baseline Fits Manager instance with parameters.

        Needed parameters
        :param config: The configuration dictionary.
        '''
        super().__init__(config)

        # Dataloaders
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        # Metric trackers
        self.train_metrics = MetricTracker(
            epoch_keys=[m.__name__ for m in self.metrics],
            iter_keys=[], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            epoch_keys=[m.__name__ for m in self.metrics],
            iter_keys=[], writer=self.writer
        )
        self.test_metrics = MetricTracker(
            iter_keys=[m.__name__ for m in self.metrics],
            epoch_keys=[], writer=self.writer
        )

    def _fit(self, model_name):
        '''
        Fitting logic for a model.

        :param model_name: Current model name.
        '''
        # Reset train metric tracker
        self.train_metrics.reset()

        # Concatenate the data and targets from train dataloader
        genomics = []
        clinicals = []
        targets = []
        survival_times = []
        vital_statuses = []
        for data, target in self.train_data_loader:
            genomic, clinical, _ = data
            target, survival_time, vital_status = target
            genomics.append(genomic)
            clinicals.append(clinical)
            targets.append(target)
            survival_times.append(survival_time)
            vital_statuses.append(vital_status)
        genomics = torch.cat(genomics)
        clinicals = torch.cat(clinicals)
        targets = torch.cat(targets)
        survival_times = torch.cat(survival_times)
        vital_statuses = torch.cat(vital_statuses)

        if model_name in ['Support_Vector_Machine']:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.models[model_name] = make_pipeline(StandardScaler(), self.models[model_name])

        # Concatenate the input data
        data = torch.hstack([genomics, clinicals])

        # Fit the model
        self.models[model_name].fit(data.numpy(), targets.numpy())

        # Get the model output
        classes = self.models[model_name].classes_
        outputs = torch.zeros(len(targets), max(classes) + 1)
        if model_name in ['Support_Vector_Machine']:
            if len(classes) > 2:
                outputs[:, classes] = torch.from_numpy(
                    self.models[model_name].decision_function(
                        data.numpy()
                    ).astype('float32')
                )
            else:
                outputs[:, classes[1:]] = torch.from_numpy(
                    self.models[model_name].decision_function(
                        data.numpy()
                    ).astype('float32').reshape(*outputs[:, classes[1:]].shape)
                )
        else:
            outputs[:, classes] = torch.from_numpy(
                self.models[model_name].predict_proba(
                    data.numpy()
                ).astype('float32')
            )

        # Update train metric tracker
        for metric in self.metrics:
            if metric.__name__ == 'c_index':
                self.train_metrics.epoch_update(metric.__name__, metric(outputs, survival_times, vital_statuses))
            else:
                self.train_metrics.epoch_update(metric.__name__, metric(outputs, targets))
        log = {'train_' + k: v for k, v in self.train_metrics.result().items()}

        # Validation
        if self.valid_data_loader:
            valid_log = self._predict_proba(model_name)
            log.update(**{'valid_' + k: v for k, v in valid_log.items()})

        # Testing
        if self.test_data_loader:
            test_log = self._bootstrap(model_name)
            log.update(**{'bootstrap_' + k: v for k, v in test_log.items()})

        return log

    def _predict_proba(self, model_name):
        '''
        Predicting logic for a model.

        :param model_name: Current model name.
        '''
        # Reset valid metric tracker
        self.valid_metrics.reset()

        # Concatenate the data and targets from valid dataloader
        genomics = []
        clinicals = []
        targets = []
        survival_times = []
        vital_statuses = []
        for data, target in self.valid_data_loader:
            genomic, clinical, _ = data
            target, survival_time, vital_status = target
            genomics.append(genomic)
            clinicals.append(clinical)
            targets.append(target)
            survival_times.append(survival_time)
            vital_statuses.append(vital_status)
        genomics = torch.cat(genomics)
        clinicals = torch.cat(clinicals)
        targets = torch.cat(targets)
        survival_times = torch.cat(survival_times)
        vital_statuses = torch.cat(vital_statuses)

        # Concatenate the input data
        data = torch.hstack([genomics, clinicals])

        # Get the model output
        classes = self.models[model_name].classes_
        outputs = torch.zeros(len(targets), max(classes) + 1)
        if model_name in ['Support_Vector_Machine']:
            if len(classes) > 2:
                outputs[:, classes] = torch.from_numpy(
                    self.models[model_name].decision_function(
                        data.numpy()
                    ).astype('float32')
                )
            else:
                outputs[:, classes[1:]] = torch.from_numpy(
                    self.models[model_name].decision_function(
                        data.numpy()
                    ).astype('float32').reshape(*outputs[:, classes[1:]].shape)
                )
        else:
            outputs[:, classes] = torch.from_numpy(
                self.models[model_name].predict_proba(
                    data.numpy()
                ).astype('float32')
            )

        # Update valid metric tracker
        for metric in self.metrics:
            if metric.__name__ == 'c_index':
                self.valid_metrics.epoch_update(metric.__name__, metric(outputs, survival_times, vital_statuses))
            else:
                self.valid_metrics.epoch_update(metric.__name__, metric(outputs, targets))

        return self.valid_metrics.result()

    def _bootstrap(self, model_name, repeat_times=1000):
        '''
        Predicting logic for a model with bootstrapping.

        :param model_name: Current model name.
        '''
        # Reset test metric tracker
        self.test_metrics.reset()

        # Start bootstrapping
        bootstrap_status = {
            'genomic': [],
            'clinical': [],
            'index': [],
            'output': [],
            'target': [],
            'survival_time': [],
            'vital_status': []
        }
        for idx in range(repeat_times):
            # Concatenate the data and targets from test dataloader
            genomics = []
            clinicals = []
            indices = []
            targets = []
            survival_times = []
            vital_statuses = []
            for data, target in self.test_data_loader:
                genomic, clinical, index = data
                target, survival_time, vital_status = target
                genomics.append(genomic)
                clinicals.append(clinical)
                indices.append(index)
                targets.append(target)
                survival_times.append(survival_time)
                vital_statuses.append(vital_status)
            genomics = torch.cat(genomics)
            clinicals = torch.cat(clinicals)
            indices = torch.cat(indices)
            targets = torch.cat(targets)
            survival_times = torch.cat(survival_times)
            vital_statuses = torch.cat(vital_statuses)

            # Concatenate the input data
            data = torch.hstack([genomics, clinicals])

            # Get the model output
            classes = self.models[model_name].classes_
            outputs = torch.zeros(len(targets), max(classes) + 1)
            if model_name in ['Support_Vector_Machine']:
                if len(classes) > 2:
                    outputs[:, classes] = torch.from_numpy(
                        self.models[model_name].decision_function(
                            data.numpy()
                        ).astype('float32')
                    )
                else:
                    outputs[:, classes[1:]] = torch.from_numpy(
                        self.models[model_name].decision_function(
                            data.numpy()
                        ).astype('float32').reshape(*outputs[:, classes[1:]].shape)
                    )
            else:
                outputs[:, classes] = torch.from_numpy(
                    self.models[model_name].predict_proba(
                        data.numpy()
                    ).astype('float32')
                )

            # Update test metric tracker
            for metric in self.metrics:
                if metric.__name__ == 'c_index':
                    self.test_metrics.iter_update(metric.__name__, metric(outputs, survival_times, vital_statuses))
                else:
                    self.test_metrics.iter_update(metric.__name__, metric(outputs, targets))

            # Record a bootstrap status
            bootstrap_status['genomic'].append(genomics.numpy())
            bootstrap_status['clinical'].append(clinicals.numpy())
            bootstrap_status['index'].append(indices.numpy())
            bootstrap_status['output'].append(outputs.numpy())
            bootstrap_status['target'].append(targets.numpy())
            bootstrap_status['survival_time'].append(survival_times.numpy())
            bootstrap_status['vital_status'].append(vital_statuses.numpy())

        # Save bootstrap statuses
        self._save_bootstrap_status(self.test_data_loader.dataset.project_id, model_name, bootstrap_status)

        return self.test_metrics.result()
