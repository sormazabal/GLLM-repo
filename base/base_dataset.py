import numpy as np
from datetime import datetime
from pathlib import Path
from abc import abstractmethod
from torch.utils.data import Dataset
from utils.logger import get_logger
from utils.util import check_cache_files


class BaseDataset(Dataset):
    '''
    Base class for all datasets
    '''
    def __init__(self, data_root_directory, cache_root_directory):
        '''
        Initialize the Base Dataset instance with parameters.

        Needed parameters
        :param data_root_dir: Specify the root directory for the downloaded files.
        :param cache_root_dir: Specify the root directory for the cache files.
        '''
        self._data_root_directory = Path(data_root_directory)
        self._cache_root_directory = Path(cache_root_directory)

        # Logger
        self.logger = get_logger('preprocess.base_dataset')

    @abstractmethod
    def __getitem__(self, index):
        '''
        Support the indexing of the dataset
        '''
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        '''
        Return the size of the dataset
        '''
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self):
        '''
        Return the input data
        '''
        raise NotImplementedError

    @property
    @abstractmethod
    def targets(self):
        '''
        Return the output data.
        '''
        raise NotImplementedError

    @property
    @abstractmethod
    def data_root_directory(self):
        '''
        Return the data directory
        '''
        return self._data_root_directory

    @property
    @abstractmethod
    def cache_root_directory(self):
        '''
        Return the cache directory
        '''
        return self._cache_root_directory

    @property
    def disease_specific_survivals(self):
        '''
        Return the 5 year disease specific survival data.
        '''
        return self._dss  # Changed from self._disease_specific_survivals

    def save_indices(self, indices):
        '''
        Saving indices

        :param indices: a dictionary of train indices and test indices
        '''
        np.savez(self.cache_root_directory.joinpath(f'indices_{datetime.now().strftime("%Y%m%d%H%M%S")}.npz'), **indices)
        self.logger.info('Saving train and test indices to {}'.format(self.cache_root_directory))

    def load_indices(self):
        '''
        Load indices
        '''
        indices_latest_file_path = check_cache_files(
            cache_directory=self.cache_root_directory,
            regex=f'indices_*'
        )

        if indices_latest_file_path:
            indices_latest_file_created_date = indices_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using indices cache files created at {} from {}'.format(
                datetime.strptime(indices_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.cache_root_directory
            ))

            indices_cache = np.load(indices_latest_file_path)
            
            return dict(indices_cache)
        else:
            return None
