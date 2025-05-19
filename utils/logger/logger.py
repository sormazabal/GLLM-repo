import logging
import logging.config
from pathlib import Path
from ..util import read_json


LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def setup_logging(log_dir, log_config='utils/logger/logger_config.json', log_level='DEBUG'):
    '''
    Setup logging configuration
    '''
    log_config = Path(log_config)
    log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_config.is_file():
        config = read_json(log_config)

        # Modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = log_dir.joinpath(handler['filename'])

        logging.config.dictConfig(config)
    else:
        logging.warning('Warning: logging configuration file is not found in {}.'.format(log_config))
        logging.basicConfig(level=LOG_LEVELS[log_level])


def get_logger(name, log_level='DEBUG'):
    message_log_level = 'Log level {} is invalid. Valid options are {}.'.format(log_level, LOG_LEVELS.keys())
    assert log_level in LOG_LEVELS, message_log_level
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS[log_level])
    return logger
