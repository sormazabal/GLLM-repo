from __future__ import annotations

import numpy as np
import torch
from dgl import batch as dgl_batch
from sklearn.model_selection import StratifiedKFold, train_test_split

from base import BaseDatasetsManager
from dataset import TCGA_Program_Dataset
from utils.logger import get_logger
from .sampler import BootstrapSubsetSampler, SubsetSampler, SubsetWeightedRandomSampler
from torch_geometric.data import Batch as PyGBatch


# def gcollate(data_list):
#     # Unzip the data_list into two lists containing the two types of tuples
#     graph_data_list, target_data_list = zip(*data_list)
#     # Unzip each list of tuples into separate lists
#     graphs, clinicals, indices, project_ids = zip(*graph_data_list)
#     targets, survival_times, vital_statuses = zip(*target_data_list)
#     # Detect if the graphs are PyG or DGL objects
#     # This checks if the first graph in the list has PyG-specific attributes
#     if hasattr(graphs[0], 'x') and hasattr(graphs[0], 'edge_index'):
#         # PyG graph object
        
#         batched_graphs = Batch.from_data_list(graphs)
#     else:
#         # DGL graph object
       
#         batched_graphs = batch(graphs)
#     batch_clinicals = torch.stack([torch.from_numpy(clinical) for clinical in clinicals])
#     batch_indices = torch.tensor(indices)
#     batch_project_ids = torch.tensor(project_ids)
#     batch_targets = torch.tensor(targets)
#     batch_survival_times = torch.tensor(survival_times)
#     batch_vital_statuses = torch.tensor(vital_statuses)
#     return ((batched_graphs, batch_clinicals, batch_indices, batch_project_ids),
#             (batch_targets, batch_survival_times, batch_vital_statuses))


def gcollate(batch):
    genomics, clinicals, indices, project_ids = zip(*[x[0] for x in batch])
    targets, survival_times, vital_statuses = zip(*[x[1] for x in batch])
    
    # Check if we're dealing with PyG graphs
    if hasattr(genomics[0], 'x') and hasattr(genomics[0], 'edge_index'):
        # Batch PyG graphs using PyG's Batch class
        batched_genomics = PyGBatch.from_data_list(list(genomics))
    else:
        # Handle DGL graphs or other data types
        batched_genomics = dgl_batch(genomics)
    
    # Convert lists to numpy arrays first
    clinicals_array = np.array(clinicals) if all(isinstance(x, np.ndarray) for x in clinicals) else clinicals
    targets_array = np.array(targets) if all(isinstance(x, (int, float, np.number)) for x in targets) else targets
    survival_times_array = np.array(survival_times) if all(isinstance(x, (int, float, np.number)) for x in survival_times) else survival_times
    vital_statuses_array = np.array(vital_statuses) if all(isinstance(x, (int, float, np.number)) for x in vital_statuses) else vital_statuses
    
    # Convert indices to tensor if it's a list of integers
    if all(isinstance(x, int) for x in indices):
        indices = torch.tensor(indices)
    
    # Convert project_ids to tensor - handle numpy values too
    if all(isinstance(x, (int, np.integer)) for x in project_ids):
        # For numeric types, convert to tensor directly
        project_ids = torch.tensor([int(p) for p in project_ids])
    elif all(isinstance(x, str) for x in project_ids):
        # For string types, convert to integers first
        project_ids = torch.tensor([int(p) for p in project_ids])
    elif all(torch.is_tensor(x) for x in project_ids):
        # For tensors, stack them
        project_ids = torch.stack(project_ids)
    else:
        # Keep as list for any other type
        project_ids = list(project_ids)
    
    return (batched_genomics, torch.tensor(clinicals_array), indices, project_ids), \
        (torch.tensor(targets_array), torch.tensor(survival_times_array), torch.tensor(vital_statuses_array))

def debug_collate(batch):
    """Debug version of gcollate that prints types and shapes"""
    import sys
    
    genomics, clinicals, indices, project_ids = zip(*[x[0] for x in batch])
    targets, survival_times, vital_statuses = zip(*[x[1] for x in batch])
    
    # Print types and shapes for debugging
    print(f"Types in collate function:")
    print(f"  genomics[0]: {type(genomics[0])}")
    print(f"  clinicals[0]: {type(clinicals[0])}, shape: {np.array(clinicals[0]).shape if hasattr(clinicals[0], 'shape') else 'unknown'}")
    print(f"  indices[0]: {type(indices[0])}")
    print(f"  project_ids[0]: {type(project_ids[0])}")
    print(f"  targets[0]: {type(targets[0])}")
    print(f"  survival_times[0]: {type(survival_times[0])}")
    print(f"  vital_statuses[0]: {type(vital_statuses[0])}")
    sys.stdout.flush()
    
    # Then continue with the regular collate function
    return gcollate(batch)


class TCGA_Datasets_Manager(BaseDatasetsManager):
    '''
    A TCGA Datasets Manager
    '''
    def __init__(self, config, datasets: dict[str, TCGA_Project_Dataset | TCGA_Program_Dataset]):
        '''
        Initialize the TCGA Datasets Manager instance with parameters.

        Needed parameters
        :param datasets: The datasets managed by this datasets manager.
        :param config: The configuration dictionary.
        '''
        self.base_datasets_manager_init_kwargs = {
            'config': config,
            'datasets': datasets
        }
        if any([dataset.graph_dataset for dataset in datasets.values()]):
            self.base_datasets_manager_init_kwargs['collate_fn'] = gcollate
            #self.base_datasets_manager_init_kwargs['collate_fn'] = debug_collate

        self.logger = get_logger('preprocess.tcga_datasets_manager')
        self.logger.info('Initializing a TCGA Datasets Manager containing {} Datasets...'.format(len(datasets)))

        super().__init__(**self.base_datasets_manager_init_kwargs)

    @staticmethod
    def get_samplers(dataset: TCGA_Project_Dataset | TCGA_Program_Dataset, test_split: float):
        indices = dataset.load_indices()

        if indices:
            if len(indices['train']) + len(indices['test']) == len(dataset):
                train_indices, test_indices = indices['train'], indices['test']
            else:
                raise ValueError('Wrong indices file')
        else:
            total_indices = np.arange(len(dataset))
            train_indices, test_indices = train_test_split(
                total_indices,
                test_size=test_split,
                stratify=dataset.stratified_targets
            )

            indices = {
                'train': train_indices,
                'test': test_indices
            }
            dataset.save_indices(indices)

        train_sampler_method = SubsetSampler
        test_sampler_method = BootstrapSubsetSampler

        samplers = {
            'train': train_sampler_method(train_indices),
            'test': test_sampler_method(test_indices, replacement=True)
        }

        return samplers

    @staticmethod
    def get_kfold_samplers(dataset: TCGA_Project_Dataset | TCGA_Program_Dataset, num_folds: int):
        '''
        Get KFold samplers.
        '''
        kfold_method = StratifiedKFold
        train_sampler_method = SubsetSampler
        valid_sampler_method = BootstrapSubsetSampler

        KFold = kfold_method(n_splits=num_folds)

        kfold_samplers = {}
        for fold, (train_indices, valid_indices) in enumerate(KFold.split(X=dataset.data, y=dataset.targets)):
            split_samplers = {
                'train': train_sampler_method(train_indices),
                'valid': valid_sampler_method(valid_indices, replacement=False),
            }

            kfold_samplers[fold] = split_samplers

        return kfold_samplers

    @staticmethod
    def get_kfold_test_samplers(dataset: TCGA_Project_Dataset | TCGA_Program_Dataset,
                                test_split: float, num_folds: int):
        '''
        Get KFold samplers, and a test sampler.
        '''
        indices = dataset.load_indices()

        if indices:
            if len(indices['train']) + len(indices['test']) == len(dataset):
                train_valid_indices, test_indices = indices['train'], indices['test']
            else:
                raise ValueError('Wrong indices file')
        else:
            total_indices = np.arange(len(dataset))
            train_valid_indices, test_indices = train_test_split(
                total_indices,
                test_size=test_split,
                stratify=dataset.stratified_targets,
            )

            indices = {
                'train': train_valid_indices,
                'test': test_indices
            }
            dataset.save_indices(indices)

        kfold_method = StratifiedKFold
        train_sampler_method = SubsetSampler
        valid_sampler_method = BootstrapSubsetSampler
        test_sampler_method = BootstrapSubsetSampler

        KFold = kfold_method(n_splits=num_folds)
        kflod_split = KFold.split(X=dataset.data[train_valid_indices], y=dataset.targets[train_valid_indices])

        kfold_samplers = {
            'train': train_sampler_method(train_valid_indices),
            'test': test_sampler_method(test_indices, replacement=True)
        }
        for fold, (train_indices, valid_indices) in enumerate(kflod_split):
            split_samplers = {
                'train': train_sampler_method(train_valid_indices[train_indices]),
                'valid': valid_sampler_method(train_valid_indices[valid_indices], replacement=False),
            }

            kfold_samplers[fold] = split_samplers

        return kfold_samplers


class TCGA_Balanced_Datasets_Manager(BaseDatasetsManager):
    '''
    A TCGA Balanced Datasets Manager
    '''
    def __init__(self, config, datasets: dict[str, TCGA_Project_Dataset | TCGA_Program_Dataset]):
        '''
        Initialize the TCGA Balanced Datasets Manager instance with parameters.

        Needed parameters
        :param datasets: The datasets managed by this datasets manager.
        :param config: The configuration dictionary.
        '''
        self.base_datasets_manager_init_kwargs = {
            'config': config,
            'datasets': datasets
        }
        if any([dataset.graph_dataset for dataset in datasets.values()]):
            self.base_datasets_manager_init_kwargs['collate_fn'] = gcollate

        self.logger = get_logger('preprocess.tcga_balanced_datasets_manager')
        self.logger.info(f'Initializing a TCGA Balanced Datasets Manager containing {len(datasets)} Datasets...')

        super().__init__(**self.base_datasets_manager_init_kwargs)

    @staticmethod
    def get_samplers(dataset: TCGA_Project_Dataset | TCGA_Program_Dataset, test_split: float):
        '''
        Get samplers.
        '''
        indices = dataset.load_indices()

        if indices:
            if len(indices['train']) + len(indices['test']) == len(dataset):
                train_indices, test_indices = indices['train'], indices['test']
            else:
                raise ValueError('Wrong indices file')
        else:
            total_indices = np.arange(len(dataset))
            train_indices, test_indices = train_test_split(
                total_indices,
                test_size=test_split,
                stratify=dataset.stratified_targets,
            )

            indices = {
                'train': train_indices,
                'test': test_indices
            }
            dataset.save_indices(indices)

        train_sampler_method = SubsetWeightedRandomSampler
        test_sampler_method = BootstrapSubsetSampler

        train_weights = dataset.weights[train_indices]

        samplers = {
            'train': train_sampler_method(train_indices, train_weights),
            'test': test_sampler_method(test_indices, replacement=True)
        }

        return samplers

    @staticmethod
    def get_kfold_samplers(dataset: TCGA_Project_Dataset | TCGA_Program_Dataset, num_folds: int):
        '''
        Get KFold samplers.
        '''
        kfold_method = StratifiedKFold
        train_sampler_method = SubsetWeightedRandomSampler
        valid_sampler_method = BootstrapSubsetSampler

        KFold = kfold_method(n_splits=num_folds, shuffle=True, random_state=0)

        kfold_samplers = {}
        for fold, (train_indices, valid_indices) in enumerate(KFold.split(X=dataset.data, y=dataset.targets)):
            train_weights = dataset.weights[train_indices]

            split_samplers = {
                'train': train_sampler_method(train_indices, train_weights),
                'valid': valid_sampler_method(valid_indices, replacement=False),
            }

            kfold_samplers[fold] = split_samplers

        return kfold_samplers

    @staticmethod
    def get_kfold_test_samplers(dataset: TCGA_Project_Dataset | TCGA_Program_Dataset, test_split: float,
                                num_folds: int):
        '''
        Get KFold samplers, and a test sampler.
        '''
        indices = dataset.load_indices()

        if indices:
            if len(indices['train']) + len(indices['test']) == len(dataset):
                train_valid_indices, test_indices = indices['train'], indices['test']
            else:
                raise ValueError('Wrong indices file')
        else:
            total_indices = np.arange(len(dataset))
            train_valid_indices, test_indices = train_test_split(
                total_indices,
                test_size=test_split,
                stratify=dataset.stratified_targets
            )

            indices = {
                'train': train_valid_indices,
                'test': test_indices
            }
            dataset.save_indices(indices)

        kfold_method = StratifiedKFold
        train_sampler_method = SubsetWeightedRandomSampler
        valid_sampler_method = BootstrapSubsetSampler
        test_sampler_method = BootstrapSubsetSampler

        KFold = kfold_method(n_splits=num_folds, shuffle=True, random_state=0)
        kfold_split = KFold.split(X=dataset.data[train_valid_indices], y=dataset.targets[train_valid_indices])

        train_weights = dataset.weights[train_valid_indices]

        kfold_samplers = {
            'train': train_sampler_method(train_valid_indices, train_weights),
            'test': test_sampler_method(test_indices, replacement=True)
        }
        for fold, (train_indices, valid_indices) in enumerate(kfold_split):
            train_weights = dataset.weights[train_valid_indices[train_indices]]

            split_samplers = {
                'train': train_sampler_method(train_valid_indices[train_indices], train_weights),
                'valid': valid_sampler_method(train_valid_indices[valid_indices], replacement=False),
            }

            kfold_samplers[fold] = split_samplers

        return kfold_samplers
