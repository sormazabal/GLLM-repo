import pandas as pd

import dataset
import datasets_manager
import runner
from base import BaseTrainer
from parse_config import ConfigParser
from utils.logger import get_logger


def multi_train_bootstrap(config: ConfigParser):
    logger = get_logger('runner.multi_train_bootstrap')

    data = {project_id: config.init_obj(f'datasets.{project_id}', dataset) for project_id in config['datasets']}
    manager = getattr(datasets_manager, config['datasets_manager.type'])(datasets=data, config=config)

    project_results = {}
    for project_id in data:
        logger.info('Training Start')
        trainer_init_config = {
            'config': config,
            'train_data_loader': manager[project_id]['dataloaders']['train'],
            'test_data_loader': manager[project_id]['dataloaders']['test']
        }
        trainer: BaseTrainer = getattr(runner, config['runner.type'])(**trainer_init_config)
        result = trainer.run()

        project_results[project_id] = result
        logger.info('Training End')
    return project_results


def multi_cross_validation_bootstrap(config: ConfigParser):
    logger = get_logger('runner.multi_cross_validation_bootstrap')

    data = {project_id: config.init_obj(f'datasets.{project_id}', dataset) for project_id in config['datasets']}
    manager = getattr(datasets_manager, config['datasets_manager.type'])(datasets=data, config=config)

    project_results = {'cross_validation': {}, 'bootstrap': {}}
    for project_id in data:
        project_cross_validation_result = {}
        logger.info('Cross Validation Start')
        for fold_index in manager[project_id]['dataloaders']:
            if not isinstance(fold_index, int):
                continue
            logger.info(f'{fold_index + 1} Fold for {project_id}...')
            trainer_init_config = {
                'config': config,
                'train_data_loader': manager[project_id]['dataloaders'][fold_index]['train'],
                'valid_data_loader': manager[project_id]['dataloaders'][fold_index]['valid']
            }
            trainer: BaseTrainer = getattr(runner, config['runner.type'])(**trainer_init_config)
            result = trainer.run()
            project_cross_validation_result[fold_index] = {k: v['mean'] for k, v in result.items()}

        project_results['cross_validation'][project_id] = pd.DataFrame.from_dict(
            project_cross_validation_result,
            'index'
        ).describe().T.loc[:, ['mean', 'std']].to_dict('index')

        logger.info('Cross Validation End')
        logger.info('Bootstrapping Start')

        trainer_init_config = {
            'config': config,
            'train_data_loader': manager[project_id]['dataloaders']['train'],
            'test_data_loader': manager[project_id]['dataloaders']['test']
        }

        trainer: BaseTrainer = getattr(runner, config['runner.type'])(**trainer_init_config)
        project_bootstrap_result = trainer.run()

        project_results['bootstrap'][project_id] = project_bootstrap_result
        logger.info('Bootstrapping End')
    return project_results
