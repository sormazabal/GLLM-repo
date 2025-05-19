import argparse
from datetime import datetime
from pathlib import Path
from warnings import filterwarnings

import lightning.pytorch as pl
import pandas as pd
import torch
import yaml
from lightning.pytorch.loggers import CSVLogger
from tqdm import tqdm
from lightning.pytorch.callbacks import Callback
import time

import model
from dataset import TCGA_Program_Dataset
from datasets_manager import TCGA_Balanced_Datasets_Manager, TCGA_Datasets_Manager
from lit_models import LitFullModel, LitBayesianClassifier, LitCancerType
from utils import config_add_subdict_key, get_logger, override_n_genes, set_random_seed, setup_logging
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from fvcore.nn import parameter_count_table
import os

CUDA_LAUNCH_BLOCKING=1

SEED = 1126
set_random_seed(SEED)

class TimingCallback(Callback):
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        # Just use print instead of trying to use logger.info
        print(f"Epoch {trainer.current_epoch} completed in {epoch_time:.2f} seconds")
        # Also log to the standard Python logger which you have set up
        import logging
        logging.getLogger(Path(trainer.default_root_dir).stem).info(
            f"Epoch {trainer.current_epoch} completed in {epoch_time:.2f} seconds"
        )
        
    def get_average_epoch_time(self):
        if not self.epoch_times:
            return 0
        return sum(self.epoch_times) / len(self.epoch_times)

def main():
    # Select a config file.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to the config file.', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    
    # assert that the task dimension is the same as the number of cancer types
    assert config['models']['Task_Classifier']['task_dim'] == len(config['datasets']['project_ids']), \
        f"Task dimension {config['models']['Task_Classifier']['task_dim']} does not match the number of cancer types {len(config['datasets']['project_ids'])}."
    override_n_genes(config)                                                    # For multi-task graph models.
    config['csv_logger'] = True if 'csv_logger' in config and config['csv_logger'] else False
    config_name = Path(args.config).stem

    # Setup logging.
    setup_logging(log_path := f'Logs/{config_name}/{datetime.now():%Y-%m-%dT%H:%M:%S}/')
    logger = get_logger(config_name)
    logger.info(f'Using Random Seed {SEED} for this experiment')
    get_logger('lightning.pytorch.accelerators.cuda', log_level='WARNING')      # Disable cuda logging.
    filterwarnings('ignore', r'.*Skipping val loop.*')                          # Disable val loop warning.

    # Create dataset manager.
    data = {'TCGA_BLC': TCGA_Program_Dataset(**config['datasets'])}
    if 'TCGA_Balanced_Datasets_Manager' == config['datasets_manager']['type']:
        manager = TCGA_Balanced_Datasets_Manager(datasets=data, config=config_add_subdict_key(config))
    else:
        manager = TCGA_Datasets_Manager(datasets=data, config=config_add_subdict_key(config))

    # Cross validation.
    validation_results = []
    for key, values in manager['TCGA_BLC']['dataloaders'].items():
        if isinstance(key, int) and config['cross_validation']:
            models, optimizers = create_models_and_optimizers(config)
            lit_model = LitFullModel(models, optimizers, config)
            trainer = pl.Trainer(                                               # Create sub-folders for each fold.
                default_root_dir=log_path,
                max_epochs=config['max_epochs'],
                log_every_n_steps=1,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=CSVLogger(save_dir=log_path, version=key) if config['csv_logger'] else True,
            )
            trainer.fit(lit_model, train_dataloaders=values['train'], val_dataloaders=values['valid'])
            if config['csv_logger']:
                os.makedirs(f'{log_path}/lightning_logs', exist_ok=True)
                validation_results.append(pd.read_csv(f'{log_path}/lightning_logs/version_{key}/metrics.csv'))
        elif key == 'train':
            train = values
        elif key == 'test':
            test = values
    if validation_results:
        df_valid_results = pd.concat(validation_results).groupby('epoch').last().drop(columns=['step'])
        logger.info('\n' + results_to_markdown_table(df_valid_results, config, 'validation'))

    # Train the final model.
    models, optimizers = create_models_and_optimizers(config)
    lit_model = LitFullModel(models, optimizers, config)
    timing_callback = TimingCallback()  # Create an instance of your callback
    trainer = pl.Trainer(
        default_root_dir=log_path,
        max_epochs=config['max_epochs'],
        enable_progress_bar=False,
        log_every_n_steps=1,
        logger=True,
        callbacks=[timing_callback],  # Add the callback here
    )

    start_time = time.time()
    trainer.fit(lit_model, train_dataloaders=train)
    training_time = time.time() - start_time

    # Use your callback for timing information
    avg_epoch_time = timing_callback.get_average_epoch_time()

    logger.info(f"Training Time: {training_time:.2f} seconds")
    logger.info(f"Average time per epoch: {avg_epoch_time:.2f} seconds")

    # print(parameter_count_table(model))

    # Test the final model.
    bootstrap_results = []
    inference_start_time = time.time()
    bootstrap_times = []

    for _ in tqdm(range(config['bootstrap_repeats']), desc='Bootstrapping'):
        iter_start = time.time()
        bootstrap_results.append(trainer.test(lit_model, dataloaders=test, verbose=False)[0])
        bootstrap_times.append(time.time() - iter_start)

    inference_time = time.time() - inference_start_time
    avg_inference_time = sum(bootstrap_times) / len(bootstrap_times) if bootstrap_times else 0

    # Log detailed inference timing information
    logger.info(f"Total Inference Time: {inference_time:.2f} seconds")
    logger.info(f"Average time per bootstrap iteration: {avg_inference_time:.6f} seconds")
    logger.info(f"Number of bootstrap iterations: {config['bootstrap_repeats']}")
    logger.info(f"Average samples processed per second: {len(test.dataset) / avg_inference_time:.2f}")

    df_test_results = pd.DataFrame.from_records(bootstrap_results)
    # check if the PRC is normally distributed
    for i in range(0, len(df_test_results.columns)):
        if 'PRC' in df_test_results.columns[i]:
            # get kurtois and skewness
            # kurtosis = df_test_results.iloc[:, i].kurtosis()
            # skewness = df_test_results.iloc[:, i].skew()
            # logger.info(f'PRC_{i} Kurtosis: {kurtosis}, Skewness: {skewness}')
            # save the PRC values to a file
            os.makedirs(f'{log_path}/lightning_logs', exist_ok=True)
            df_test_results.iloc[:, i].to_csv(f'{log_path}/lightning_logs/PRC_{i}.csv', index=False)

    logger.info('\n' + results_to_markdown_table(df_test_results, config, 'test'))


def create_models_and_optimizers(config: dict):
    models: dict[str, torch.nn.Module] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}

    for model_name, kargs in config['models'].items():
        if 'Extractor' in model_name:
            models['feat_ext'] = getattr(model, model_name)(**kargs)
        elif 'Classifier' in model_name:
            models['clf'] = getattr(model, model_name)(**kargs)
        else:
            raise ValueError(f'Unknown model type: {model_name}')

    for key, optim_dict in config['optimizers'].items():
        opt_name = next(iter(optim_dict))
        if key == 'all':
            params = [param for m in models.values() for param in m.parameters()]
            optimizers[key] = getattr(torch.optim, opt_name)(params, **optim_dict[opt_name])
        else:
            optimizers[key] = getattr(torch.optim, opt_name)(models[key].parameters(), **optim_dict[opt_name])
    return models, optimizers


def results_to_markdown_table(df_results: pd.DataFrame, config: dict, mode: str) -> str:
    describe = df_results.describe()
    metrics = list(dict.fromkeys([col.split('_')[0] for col in describe.columns if 'loss' not in col]))
    losses = [col for col in describe.columns if 'loss' in col]

    table = pd.DataFrame(columns=[mode] + metrics + losses)
    table[mode] = config['datasets']['project_ids'] + ['all'] if losses else ''
    table.set_index(mode, inplace=True)

    for metric in metrics:
        for project_id, project in enumerate(config['datasets']['project_ids']):
            col = f'{metric}_{project_id}'
            assert col in describe.columns, f'{col} not in {describe.columns} when summarizing {mode} results.'
            table.loc[project, metric] = f'{describe.loc["mean", col]:.5f} ± {describe.loc["std", col]:.5f}'
    for loss in losses:
        assert loss in describe.columns, f'{loss} not in {describe.columns} when summarizing {mode} results.'
        table.loc['all', loss] = f'{describe.loc["mean", loss]:.5f} ± {describe.loc["std", loss]:.5f}'
    return table.to_markdown(tablefmt='pipe', stralign='right', numalign='right')


if __name__ == '__main__':
    main()
