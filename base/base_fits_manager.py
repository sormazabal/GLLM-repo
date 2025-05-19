from abc import abstractmethod
from pathlib import Path

import numpy as np

import model.baseline as model
import utils.runner.metric as module_metric
from parse_config import ConfigParser
from utils.logger import TensorboardWriter, get_logger


class BaseFitsManager(object):
    '''
    Base class for all fits managers
    '''
    def __init__(self, config: ConfigParser):
        '''
        Initialize the Base Fits Manager instance with parameters.

        Needed parameters
        :param config: The configuration dictionary.
        '''
        self.config = config
        self.logger = get_logger('runner.base_fits_manager')

        # Create models
        self.models = {
            model_name: self.config.init_obj(f'models.{model_name}', model) for model_name in self.config['models']
        }

        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.ckpt_dir)

        # Metric functions
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        # Setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, config['runner.tensorboard'])

    @abstractmethod
    def _fit(self, model_name):
        '''
        Fitting logic for a model.

        :param model_name: Current model name.
        '''
        raise NotImplementedError

    def run(self):
        '''
        Full fitting logic
        '''
        results = {}

        for model_name in self.models:
            result = self._fit(model_name)

            results.update({f'{model_name}_{k}': v for k, v in result.items()})

            # Save logged informations into log dict
            log = {'model_name': model_name}
            log.update(result)

            # Log informations
            for key, value in log.items():
                if isinstance(value, dict):
                    self.logger.info('{:20s}: {:.5f} ±{:.5f}'.format(str(key).lower(), value['mean'], value['std']))
                else:
                    self.logger.info('{:20s}: {}'.format(str(key).lower(), value))

        return results

    def _save_bootstrap_status(self, project_id, model_name, bootstrap_status):
        '''
        Save bootstrap statuses

        :param bootstrap_statuses:
        '''
        for k, v in bootstrap_status.items():
            bootstrap_status[k] = np.stack(v, axis=0)

        project_id = project_id.lower()
        model_name = model_name.lower()

        np.savez(self.config.log_dir.joinpath(f'{project_id}_{model_name}_bootstrap_status.npz'), **bootstrap_status)
        self.logger.info('Saving bootstrap status to {}'.format(self.config.log_dir))
