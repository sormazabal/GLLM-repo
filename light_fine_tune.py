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

import model
from dataset import TCGA_Program_Dataset
from datasets_manager import TCGA_Balanced_Datasets_Manager, TCGA_Datasets_Manager
from lit_models import LitFullModel, LitFineTuning
from utils import config_add_subdict_key, get_logger, override_n_genes, set_random_seed, setup_logging

SEED = 1126
set_random_seed(SEED)


def main():
    # Select a config file.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to the config file.', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
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
            pretrained_genomic = LitFullModel.load_from_checkpoint(
                config['pretrain']['genomic'], strict=False, models=models, optimizers=optimizers, config=config
            )
            pretrained_clinical = LitFullModel.load_from_checkpoint(
                config['pretrain']['clinical'], strict=False, models=models, optimizers=optimizers, config=config
            )
            lit_model = LitFineTuning(models, optimizers, config, pretrained_genomic, pretrained_clinical)
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
    pretrained_genomic = LitFullModel.load_from_checkpoint(
        config['pretrain']['genomic'], strict=False, models=models, optimizers=optimizers, config=config
    )
    pretrained_clinical = LitFullModel.load_from_checkpoint(
        config['pretrain']['clinical'], strict=False, models=models, optimizers=optimizers, config=config
    )
    lit_model = LitFineTuning(models, optimizers, config, pretrained_genomic, pretrained_clinical)
    trainer = pl.Trainer(
        default_root_dir=log_path,
        max_epochs=config['max_epochs'],
        enable_progress_bar=False,
        log_every_n_steps=1,
        logger=False,
    )
    trainer.fit(lit_model, train_dataloaders=train)

    # Test the final model.
    bootstrap_results = []
    for _ in tqdm(range(config['bootstrap_repeats']), desc='Bootstrapping'):
        bootstrap_results.append(trainer.test(lit_model, dataloaders=test, verbose=False)[0])
    df_test_results = pd.DataFrame.from_records(bootstrap_results)
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
