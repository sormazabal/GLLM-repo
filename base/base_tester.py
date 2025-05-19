import torch
from abc import abstractmethod
from utils.logger import get_logger
import model as module_model
import utils.runner.loss as module_loss
import utils.runner.metric as module_metric
import utils.runner.plot as module_plot


class BaseTester(object):
    '''
    Base class for all testers
    '''
    def __init__(self, config):
        '''
        Initialize the Base Tester instance with parameters.

        Needed parameters
        :param config: The configuration dictionary.
        '''
        self.config = config
        self.logger = get_logger('tester')

        # Setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.non_blocking = config['pin_memory']
        self.models = {model_name :self.config.init_obj(f'models.{model_name}', module_model).to(self.device) for model_name in self.config['models']}
        if len(device_ids) > 1:
            for model_name in self.models:
                self.models[model_name] = torch.nn.parallel.DistributedDataParallel(self.models[model_name], device_ids=device_ids)

        # Loss functions
        self.losses = {loss: config.init_obj(f'losses.{loss}', module_loss) for loss in config['losses']}

        # Metric functions
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        # Plot functions
        self.plots = [getattr(module_plot, plt) for plt in config['plots']]

        # Tester parameters settings
        self.checkpoint_dir = self.config.ckpt_dir
        self.output_dir = self.config.log_dir

        # Resume from the checkpoint
        self._resume_checkpoint(config.resume)

    @abstractmethod
    def _test(self):
        '''
        Testing logic
        '''
        raise NotImplementedError

    def run(self):
        '''
        Full testing logic
        '''
        result = self._test()

        # Save logged informations into log dict
        log = result

        # Log informations
        for key, value in log.items():
            if isinstance(value, float):
                self.logger.info('{:15s}: {:.5f}'.format(str(key).lower(), value))
            else:
                self.logger.info('{:15s}: {}'.format(str(key).lower(), value))

    def _prepare_device(self, n_gpu_use):
        '''
        Setup GPU device if available, move model into configured device
        '''
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning('There\'s no GPU available on this machine, training will be performed on CPU.')
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning('The number of GPU\'s configured to use is {}, but only {} are available on this machine.'.format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _resume_checkpoint(self, resume_path):
        '''
        Resume from checkpoint

        :param resume_path: Checkpoint path to be resumed
        '''
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {}...".format(resume_path))
        checkpoint = torch.load(resume_path)

        #TODO
        # Check the models type
        for model_name in self.models:
            self.models[model_name].load_state_dict(checkpoint['models'][model_name])

        self.logger.info('Checkpoint loaded')

        filename = str(self.checkpoint_dir / 'checkpoint-test.pth')
        torch.save(checkpoint, filename)
        self.logger.info('Saving checkpoint...')
