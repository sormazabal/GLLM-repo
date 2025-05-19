import re
import os
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from utils.logger import setup_logging
from utils.util import read_json, read_yaml, write_json


class ConfigParser(object):
    def __init__(self, config, resume=None, modification=None, run_id=None):
        '''
        Class to parse configuration yaml file. Handles hyperparameters for training, initializations of modules and initial logging module.

        Needed parameters
        :param config: Dictionary containing configurations, hyperparameters for training. Contents of `config.yaml` file for example.

        Optional parameters
        :param resume: Path to the checkpoint being loaded.
        :param modification: Dictionary specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default.
        '''
        # Load config file and apply modification
        self._config = self._update_config(config, modification)
        self._resume = resume

        experiment_name = self.config['name']
        # Use timestamp as default run_id
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # Checkpoint directory
        self._ckpt_dir = Path(self.config['checkpoint_directory']).joinpath(experiment_name, run_id)

        # Log directory
        self._log_dir = Path(self.config['log_directory']).joinpath(experiment_name, run_id)

        # Make directory for saving checkpoints and log. Skipping testing
        exist_ok = re.search(r'test$', run_id)
        self.ckpt_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # Save updated config file to the checkpoint dir
        write_json(self.config, self.ckpt_dir.joinpath('config.json'))

        # Configure logging module
        setup_logging(self.log_dir)

    @classmethod
    def from_args(cls, args, options='', run_id=None):
        '''
        Initialize this class from cli arguments.

        :param args: Argument parser from input arguments.
        :param options: Custom CLI options to modify configuration from default values.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default.
        '''
        # Update custom cli options to argement parser
        for option in options:
            args.add_argument(*option.flags, default=None, type=option.type)
        args = args.parse_args()

        # Specify GPU devices
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        # Load resume configuration file if exists
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent.joinpath('config.json')
            config = read_json(cfg_fname)
        else:
            message = 'Configuration file need to be specified. Add \'-c config.yaml\', for example.'
            assert args.config is not None, message
            resume = None
            cfg_fname = Path(args.config)

            if Path(cfg_fname).suffix == '.yaml':
                config = read_yaml(cfg_fname)
            else:
                raise FileNotFoundError('Didn\'t find any yaml files')

        # Update new config from resume configuration file
        if args.config and resume:
            config.update(read_yaml(args.config))

        # Parse custom cli options into dictionary
        modification = {option.target : getattr(args, cls._get_opt_name(option.flags)) for option in options}
        return cls(config, resume, modification, run_id)

    def init_obj(self, name, module, *args, **kwargs):
        '''
        Finds a function handle with the name given as 'type' in config, and returns the instance initialized with corresponding arguments given. For example, `object = config.init_obj('name', module, a, b=1)` is equivalent to `object = module.name(a, b=1)`.

        :param name: Name of the object.
        :param module: The module used to initialize.
        '''
        module_name = self[f'{name}.type']
        module_args = dict(self[f'{name}.args'])
        message = 'Overwriting kwargs given in config file is not allowed.'
        assert all([k not in module_args for k in kwargs]), message
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        '''
        Finds a function handle with the name given as 'type' in config, and returns the function with given arguments fixed with functools.partial. For example, `function = config.init_ftn('name', module, a, b=1)` is equivalent to `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.

        :param name: Name of the function.
        :param module: The module used to call.
        '''
        module_name = self[f'{name}.type']
        module_args = dict(self[f'{name}.args'])
        message = 'Overwriting kwargs given in config file is not allowed.'
        assert all([k not in module_args for k in kwargs]), message
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        '''
        Access items like ordinary dict.

        :param name: Name of the keys.
        '''
        names = name.split('.')
        if len(names) <= 1:
            return self.config[name]
        else:
            return self._get_by_path(self.config, names)

    @property
    def config(self):
        '''
        Return the dictionary of the configuration.
        '''
        return self._config

    @property
    def resume(self):
        '''
        Return the resume path
        '''
        if self._resume is not None:
            return Path(self._resume)
        return self._resume

    @property
    def ckpt_dir(self):
        '''
        Return the checkpoint directory.
        '''
        return self._ckpt_dir

    @property
    def log_dir(self):
        '''
        Return the log directory.
        '''
        return self._log_dir

    @staticmethod
    def _update_config(config, modification):
        '''
        Helper functions to update config dict with custom cli options

        :param config: Dictionary containing configurations, hyperparameters for training. Contents of `config.yaml` file for example.
        :param modification: Dictionary specifying position values to be replaced from config dict.
        '''
        if modification is None:
            return config

        for key, value in modification.items():
            if value is not None:
                keys = key.split('.')
                ConfigParser._set_by_path(config, keys, value)

        return config

    # TODO
    # Need to redesign this function
    @staticmethod
    def _get_opt_name(flags):
        for flg in flags:
            if flg.startswith('--'):
                return flg.replace('--', '')
        return flags[0].replace('--', '')

    @staticmethod
    def _set_by_path(tree, keys, value):
        '''
        Set a value in a nested object in tree by sequence of keys.
        '''
        ConfigParser._get_by_path(tree, keys[:-1])[keys[-1]] = value

    @staticmethod
    def _get_by_path(tree, keys):
        '''
        Access a nested object in tree by sequence of keys.
        '''
        return reduce(getitem, keys, tree)
