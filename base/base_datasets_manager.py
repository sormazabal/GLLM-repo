from abc import abstractmethod
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from utils.logger import get_logger
import random
import numpy
import torch

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class BaseDatasetsManager(object):
    '''
    Base class for all datasets managers
    '''
    def __init__(self, datasets, config, collate_fn=default_collate):
        '''
        Initialize the Base Datasets Manager instance with parameters.

        Needed parameters
        :param datasets: The datasets managed by this datasets manager.
        :param config: The configuration dictionary.

        Optional parameters
        :param collate_fn: The function that transform dataset output.
        '''
        self.config = config
        self._datasets = datasets
        self.collate_fn = collate_fn
        self._dataloaders = self._get_dataloaders()

        # Logger
        self.logger = get_logger('preprocess.base_datasets_manager')

    def __getitem__(self, index):
        '''
        Support the indexing of the dataset manager.
        '''
        return {
            'dataset': self.datasets[index],
            'dataloaders': self.dataloaders[index]
        }

    def _get_dataloaders(self):
        '''
        Get the dataloaders.
        '''
        dataloaders = {}

        for dataset_name in self.datasets:
            data_loader_init_kwargs = {
                'batch_size': self.config[f'datasets_manager.{dataset_name}.batch_size'],
                'num_workers': self.config[f'datasets_manager.{dataset_name}.num_workers'],
                'pin_memory': self.config[f'pin_memory'],
                'collate_fn': self.collate_fn
            }

            dataloader = {}
            dataset = self.datasets[dataset_name]

            # Define the behavior of batch size equals to 0
            if data_loader_init_kwargs['batch_size'] == 0:
                data_loader_init_kwargs['batch_size'] = len(dataset)

            # Get samplers
            if self.config[f'datasets_manager.{dataset_name}.train']:
                if 'test_split' in self.config[f'datasets_manager.{dataset_name}']:
                    if 'num_folds' in self.config[f'datasets_manager.{dataset_name}']:
                        samplers = self.get_kfold_test_samplers(
                            dataset=dataset,
                            test_split=self.config[f'datasets_manager.{dataset_name}.test_split'],
                            num_folds=self.config[f'datasets_manager.{dataset_name}.num_folds']
                        )
                    else:
                        samplers = self.get_samplers(
                            dataset=dataset,
                            test_split=self.config[f'datasets_manager.{dataset_name}.test_split']
                        )
                else:
                    if 'num_folds' in self.config[f'datasets_manager.{dataset_name}']:
                        samplers = self.get_kfold_samplers(
                            dataset=dataset,
                            num_folds=self.config[f'datasets_manager.{dataset_name}.num_folds']
                        )
                    else:
                        samplers = {
                            'train': None
                        }
                    
                        data_loader_init_kwargs['shuffle'] = True
            else:
                samplers = {
                    'test': None
                }

            # Build dataloaders
            if 'num_folds' in self.config[f'datasets_manager.{dataset_name}']:
                for key in samplers:
                    if isinstance(key, int):
                        fold = key
                        fold_dataloaders = {}
                        for split in samplers[fold]:
                            fold_dataloaders[split] = DataLoader(
                                dataset=dataset,
                                sampler=samplers[fold][split],
                                worker_init_fn=seed_worker,
                                **data_loader_init_kwargs
                            )

                        dataloader[fold] = fold_dataloaders
                    else:
                        split = key
                        dataloader[split] = DataLoader(
                            dataset=dataset,
                            sampler=samplers[split],
                            worker_init_fn=seed_worker,
                            **data_loader_init_kwargs
                        )

            else:
                for split in samplers:
                    dataloader[split] = DataLoader(
                        dataset=dataset,
                        sampler=samplers[split],
                        worker_init_fn=seed_worker,
                        **data_loader_init_kwargs
                    )

            dataloaders[dataset_name] = dataloader

        return dataloaders

    def __len__(self):
        '''
        Return the size of the dataset.
        '''
        return len(self.datasets)

    @staticmethod
    @abstractmethod
    def get_samplers(dataset, test_split):
        '''
        Get samplers.
        '''
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kfold_samplers(dataset, num_folds):
        '''
        Get KFold samplers.
        '''
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kfold_test_samplers(dataset, test_split, num_folds):
        '''
        Get KFold samplers, and a test sampler.
        '''
        raise NotImplementedError

    @property
    def datasets(self):
        '''
        Return all datasets.
        '''
        return self._datasets

    @property
    def dataloaders(self):
        '''
        Return all dataloaders
        '''
        return self._dataloaders
