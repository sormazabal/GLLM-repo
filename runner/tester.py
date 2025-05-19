import torch
from base import BaseTester
from .tracker import MetricTracker


class Tester(BaseTester):
    '''
    Tester
    '''
    def __init__(self, config, test_data_loader):
        '''
        Initialize the Tester instance with parameters.

        Needed parameters
        :param config: The configuration dictionary.
        '''
        super().__init__(config)

        # Dataloaders
        self.test_data_loader = test_data_loader

        # Metric trackers
        self.test_metrics = MetricTracker(
            epoch_keys=[m.__name__ for m in self.metrics],
            iter_keys=['loss']
        )

    def _test(self):
        '''
        Testing logic
        '''
        # Set models to validating mode
        for model_name in self.models:
            self.models[model_name].eval()

        # Reset test metric tracker
        self.test_metrics.reset()

        # Start testing
        outputs = []
        targets = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                genomic, clinical = data

                genomic = genomic.to(self.device, non_blocking=self.non_blocking)
                clinical = clinical.to(self.device, non_blocking=self.non_blocking)
                target = target.to(self.device, non_blocking=self.non_blocking)

                # Models' output for one batch
                output = self.models['DNN'](genomic)
                loss = self.losses['cross_entropy'](output, target)

                outputs.append(output.detach().cpu())
                targets.append(target.detach().cpu())

                self.test_metrics.iter_update('loss', loss.item())

            outputs = torch.cat(outputs)
            targets = torch.cat(targets)

            for metric in self.metrics:
                self.test_metrics.epoch_update(metric.__name__, metric(outputs, targets))

        return {'test_'+k: v for k, v in self.test_metrics.result().items()}
