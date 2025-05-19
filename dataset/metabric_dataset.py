from pathlib import Path

import numpy as np
import pandas as pd

from base import BaseDataset
from preprocess import METABRIC
from utils.logger import get_logger


class METABRIC_Dataset(BaseDataset):
    '''
    TCGA Project Dataset
    '''
    def __init__(self, data_directory, cache_directory, target_type='overall_survival'):
        '''
        Initialize the METABRIC Dataset with parameters.

        Needed parameters
        :param data_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.

        Optional parameters
        :param target_type: Specify the wanted target type that you want to use.
        '''
        self.project_id = 'METABRIC'

        # Logger
        self.logger = get_logger('preprocess.metabric_dataset')
        self.logger.info('Creating a {} Project Dataset...'.format(self.project_id))

        # Directory for data and cache files
        self.data_directory = Path(data_directory)
        self.cache_directory = Path(cache_directory)

        # Create METABRIC instance
        self.metabric_init_kwargs = {
            'project_id': 'METABRIC',
            'download_directory': data_directory,
            'cache_directory': cache_directory,
        }
        self.metabric = METABRIC(**self.metabric_init_kwargs)

        # Specify the target type
        self.target_type = target_type

        # Get data from METABRIC instance
        self._getdata()
        self.logger.info('Total {} patients, {} genomic features and {} clinical features'.format(
            len(self.targets[self.targets >= 0]), len(self.genomic_ids), len(self.clinical_ids)
        ))
        self.logger.info('Target Type {}'.format(self.target_type))
        self.logger.info('Overall survival imbalance ratio {} %'.format(
            sum(self.overall_survivals) / len(self.overall_survivals) * 100
        ))
        self.logger.info('Disease specific survival event rate {} %'.format(
            sum(self.disease_specific_survivals >= 0) / len(self.disease_specific_survivals) * 100
        ))
        self.logger.info('Disease specific survival imbalance ratio {} %'.format(
            sum(self.disease_specific_survivals[self.disease_specific_survivals >= 0]) / len(
                self.overall_survivals[self.disease_specific_survivals >= 0]
            ) * 100
        ))

        # Initialize BaseDataset instance
        self.base_dataset_init_kwargs = {
            'data_root_directory': data_directory,
            'cache_root_directory': cache_directory
        }
        super().__init__(**self.base_dataset_init_kwargs)

    def _getdata(self):
        '''
        Get the data from METABRIC
        '''
        df_genomic = self.metabric.genomic.T
        df_clinical = self.metabric.clinical.T

        # Setting copy=False will not work.
        df_clinical = df_clinical.astype({'AGE_AT_DIAGNOSIS': 'float64', 'TUMOR_SIZE': 'float64'})
        mean = df_clinical[['AGE_AT_DIAGNOSIS', 'TUMOR_SIZE']].mean()
        std = df_clinical[['AGE_AT_DIAGNOSIS', 'TUMOR_SIZE']].std()
        df_clinical[['AGE_AT_DIAGNOSIS', 'TUMOR_SIZE']] = (df_clinical[['AGE_AT_DIAGNOSIS', 'TUMOR_SIZE']] - mean) / std

        df_clinical = pd.get_dummies(df_clinical, columns=['CELLULARITY', 'RADIO_THERAPY', 'CHEMOTHERAPY',
                                                           'HISTOLOGICAL_SUBTYPE', 'HORMONE_THERAPY', 'BREAST_SURGERY',
                                                           'INFERRED_MENOPAUSAL_STATE', 'ER_STATUS', 'HER2_STATUS',
                                                           'PR_STATUS', 'TUMOR_SIZE'], dtype=float)

        df_overall_survival = self.metabric.overall_survival.T
        df_disease_specific_survival = self.metabric.disease_specific_survival.T

        df_total = pd.concat([df_genomic, df_clinical, df_overall_survival, df_disease_specific_survival],
                             axis=1, join='inner')

        self._genomics = df_total[df_genomic.columns].to_numpy()
        self._clinicals = df_total[df_clinical.columns].to_numpy()
        self._overall_survivals = df_total[df_overall_survival.columns].squeeze().to_numpy()
        self._disease_specific_survivals = df_total[df_disease_specific_survival.columns].squeeze().to_numpy()
        self._patient_ids = tuple(df_total.index.to_list())
        self._genomic_ids = tuple(df_genomic.columns.to_list())
        self._clinical_ids = tuple(df_clinical.columns.to_list())
        return

    def __getitem__(self, index):
        '''
        Support the indexing of the dataset
        '''
        return (self.genomics[index], self.clinicals[index]), self.targets[index]

    def __len__(self):
        '''
        Return the size of the dataset
        '''
        return len(self.targets)

    @property
    def data(self):
        '''
        Return the genomic and clinical data.
        '''
        return np.hstack((self._genomics, self._clinicals))

    @property
    def genomics(self):
        '''
        Return the genomic data.
        '''
        return self._genomics

    @property
    def clinicals(self):
        '''
        Return the clinical data.
        '''
        return self._clinicals

    @property
    def overall_survivals(self):
        '''
        Return the 5 year overall survival data.
        '''
        return self._overall_survivals

    @property
    def disease_specific_survivals(self):
        '''
        Return the 5 year disease specific survival data.
        '''
        return self._disease_specific_survivals

    @property
    def targets(self):
        '''
        Return the target data according to target_type.
        '''
        if self.target_type == 'overall_survival':
            return self._overall_survivals
        elif self.target_type == 'disease_specific_survival':
            return self._disease_specific_survivals
        else:
            raise KeyError('Wrong target type')

    @property
    def patient_ids(self):
        '''
        Return the patient ids
        '''
        return self._patient_ids

    @property
    def genomic_ids(self):
        '''
        Return the genomic ids
        '''
        return self._genomic_ids

    @property
    def clinical_ids(self):
        '''
        Return the clinical ids
        '''
        return self._clinical_ids
