from utils.logger import get_logger
from pathlib import Path
from datetime import datetime
import pandas as pd
from utils.util import check_cache_files

class METABRIC(object):
    '''
    METABRIC
    '''
    def __init__(self, project_id, download_directory, cache_directory):
        '''
        Initialize the METABRIC instance with parameters.

        Needed parameters
        :param download_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.
        '''
        self.project_id = project_id

        # Logger
        self.logger = get_logger('preprocess.metabric')

        # Directory for download files
        self.download_directory = Path(download_directory)
        self.download_directory.mkdir(exist_ok=True)

        # Directory for cache files
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(exist_ok=True)

        # Get initial case_ids
        self.case_ids = self._get_case_data(download_directory=self.download_directory)

        # Genomic data 
        self._genomic = self._get_genomic_data(
            case_ids=self.case_ids,
            download_directory=self.download_directory,
            cache_directory=self.cache_directory
        )

        # Clinical data
        self._clinical = self._get_clinical_data(
            case_ids=self.case_ids,
            download_directory=self.download_directory,
            cache_directory=self.cache_directory
        )

        # Survival data
        self._overall_survival = self._get_overall_survival_data(
            case_ids=self.case_ids,
            download_directory=self.download_directory,
            cache_directory=self.cache_directory
        )
        self._disease_specific_survival = self._get_disease_specific_survival_data(
            case_ids=self.case_ids,
            download_directory=self.download_directory,
            cache_directory=self.cache_directory
        )

    def _get_case_data(self, download_directory):
        '''
        Get the case data.

        :param download_directory: Specify the directory for the downloaded files.
        '''
        case_file_path = download_directory.joinpath('case_lists', 'cases_RNA_Seq_mRNA.txt')

        self.logger.info('Loading case ids for {}...'.format(self.project_id))

        case_ids = []
        with open(case_file_path, 'r') as f:
            case_ids = f.readlines()[-1].split(' ')[-1].split('\n')[0].split('\t')

        return case_ids

    def _get_genomic_data(self, case_ids, download_directory, cache_directory):
        '''
        Get the genomic data.

        :param case_ids: Specify the case ids.
        :param download_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        genomic_latest_file_path = check_cache_files(cache_directory=cache_directory, regex=f'genomic_*')

        if genomic_latest_file_path:
            genomic_latest_file_created_date = genomic_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using genomic cache files created at {} for {}'.format(
                datetime.strptime(genomic_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_genomic_cache = pd.read_csv(genomic_latest_file_path, sep='\t', index_col='gene_id')

            return df_genomic_cache

        genomic_file_path = download_directory.joinpath('data_mrna_agilent_microarray.txt')

        self.logger.info('Loading genomic data for {}...'.format(self.project_id))
        df_genomic = pd.read_csv(
            genomic_file_path,
            sep='\t', index_col='Hugo_Symbol'
        ).T.reindex(index=case_ids).dropna()
        df_genomic.index.rename(name=None, inplace=True)
        df_genomic.columns.rename(name='gene_id', inplace=True)

        df_genomic = df_genomic[['ELAVL1', 'EGFR', 'BTRC', 'FBXO6', 'SHMT2', 'KRAS', 'SRPK2', 'YWHAQ', 'PDHA1', 'EWSR1', 'ZDHHC17', 'ENO1', 'DBN1', 'PLK1', 'ESR1', 'GSK3B']]

        # Save the genomic data
        df_genomic = df_genomic.T
        df_genomic.to_csv(cache_directory.joinpath(f'genomic_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'), sep='\t')
        self.logger.info('Saving results for {} to cache file'.format(self.project_id))

        return df_genomic

    def _get_clinical_data(self, case_ids, download_directory, cache_directory):
        '''
        Get the clinical data.

        :param case_ids: Specify the case ids.
        :param download_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        clinical_latest_file_path = check_cache_files(cache_directory=cache_directory, regex='clinical_*')

        if clinical_latest_file_path:
            clinical_latest_file_created_date = clinical_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using clinical cache files created at {} for {}'.format(
                datetime.strptime(clinical_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_clinical_cache = pd.read_csv(clinical_latest_file_path, sep='\t', index_col='clinical')
            
            return df_clinical_cache

        clinical_patient_file_path = download_directory.joinpath('data_clinical_patient.txt')
        clinical_sample_file_path = download_directory.joinpath('data_clinical_sample.txt')

        self.logger.info('Loading clinical data for {}...'.format(self.project_id))

        df_clinical_patient = pd.read_csv(
            clinical_patient_file_path,
            sep='\t', skiprows=4, index_col='PATIENT_ID'
        ).reindex(index=case_ids)
        df_clinical_patient = df_clinical_patient[['AGE_AT_DIAGNOSIS', 'CELLULARITY', 'RADIO_THERAPY', 'CHEMOTHERAPY', 'HISTOLOGICAL_SUBTYPE', 'HORMONE_THERAPY', 'BREAST_SURGERY', 'INFERRED_MENOPAUSAL_STATE']].dropna()

        df_clinical_sample = pd.read_csv(
            clinical_sample_file_path,
            sep='\t', skiprows=4, index_col='PATIENT_ID'
        ).reindex(index=case_ids)
        df_clinical_sample = df_clinical_sample[['ER_STATUS', 'HER2_STATUS', 'PR_STATUS', 'TUMOR_SIZE']].dropna()

        df_clinical = pd.concat([df_clinical_patient, df_clinical_sample], axis=1, join="inner")
        df_clinical.index.rename(name=None, inplace=True)
        df_clinical.columns.rename(name='clinical', inplace=True)

        # Save the clinical data
        df_clinical = df_clinical.T
        df_clinical.to_csv(cache_directory.joinpath(f'clinical_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'), sep='\t')
        self.logger.info('Saving results for {} to cache file'.format(self.project_id))

        return df_clinical

    def _get_overall_survival_data(self, case_ids, download_directory, cache_directory):
        '''
        Get the 5 years overall survival data.

        :param case_ids: Specify the case ids.
        :param download_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        overall_survival_latest_file_path = check_cache_files(cache_directory=cache_directory, regex=f'overall_survival_*')

        if overall_survival_latest_file_path:
            overall_survival_latest_file_created_date = overall_survival_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using overall survival cache files created at {} for {}'.format(
                datetime.strptime(overall_survival_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_overall_survival_cache = pd.read_csv(overall_survival_latest_file_path, sep='\t', index_col='overall_survival')

            return df_overall_survival_cache

        clinical_patient_file_path = download_directory.joinpath('data_clinical_patient.txt')

        self.logger.info('Loading overall survival data for {}...'.format(self.project_id))

        df_overall_survival = pd.read_csv(
            clinical_patient_file_path,
            sep='\t', skiprows=4, index_col='PATIENT_ID'
        ).reindex(index=case_ids)
        df_overall_survival = df_overall_survival[['OS_STATUS', 'OS_MONTHS']].dropna()

        df_overall_survival = ~((df_overall_survival['OS_STATUS'] == '1:DECEASED') & (df_overall_survival['OS_MONTHS'] < 60))
        df_overall_survival = df_overall_survival.astype('int64')
        df_overall_survival = df_overall_survival.to_frame(name='overall_survival')
        df_overall_survival.index.rename(name=None, inplace=True)

        # Save the overall survival data
        df_overall_survival = df_overall_survival.T
        df_overall_survival.index.rename(name='overall_survival', inplace=True)
        df_overall_survival.to_csv(cache_directory.joinpath(f'overall_survival_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'), sep='\t')
        self.logger.info('Saving results for {} to cache file'.format(self.project_id))

        return df_overall_survival

    def _get_disease_specific_survival_data(self, case_ids, download_directory, cache_directory):
        '''
        Get the 5 years disease specific survival data.

        :param case_ids: Specify the case ids.
        :param download_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        disease_specific_survival_latest_file_path = check_cache_files(cache_directory=cache_directory, regex=f'disease_specific_survival_*')

        if disease_specific_survival_latest_file_path:
            disease_specific_survival_latest_file_created_date = disease_specific_survival_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using disease specific survival cache files created at {} for {}'.format(
                datetime.strptime(disease_specific_survival_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_disease_specific_survival_cache = pd.read_csv(disease_specific_survival_latest_file_path, sep='\t', index_col='disease_specific_survival')

            return df_disease_specific_survival_cache

        clinical_patient_file_path = download_directory.joinpath('data_clinical_patient.txt')

        self.logger.info('Loading disease specific survival data for {}...'.format(self.project_id))

        df_disease_specific_survival = pd.read_csv(
            clinical_patient_file_path,
            sep='\t', skiprows=4, index_col='PATIENT_ID'
        ).reindex(index=case_ids)
        df_disease_specific_survival = df_disease_specific_survival[['VITAL_STATUS', 'OS_MONTHS']].dropna()

        df_vital_status = df_disease_specific_survival['VITAL_STATUS'] == 'Died of Disease'

        df_disease_specific_survival = df_vital_status * (df_disease_specific_survival['OS_MONTHS'] > 60)
        df_disease_specific_survival = df_disease_specific_survival.astype('int64')

        df_disease_specific_survival = ~df_vital_status * -1 + df_disease_specific_survival
        df_disease_specific_survival = df_disease_specific_survival.to_frame(name='disease_specific_survival')
        df_disease_specific_survival.index.rename(name=None, inplace=True)

        # Save the disease specific survival data
        df_disease_specific_survival = df_disease_specific_survival.T
        df_disease_specific_survival.index.rename(name='disease_specific_survival', inplace=True)
        df_disease_specific_survival.to_csv(cache_directory.joinpath(f'disease_specific_survival_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'), sep='\t')
        self.logger.info('Saving results for {} to cache file'.format(self.project_id))

        return df_disease_specific_survival

    @property
    def genomic(self):
        '''
        Return the genomic data with gene_ids.
        '''
        return self._genomic

    @property
    def clinical(self):
        '''
        Return the clinical data.
        '''
        return self._clinical

    @property
    def overall_survival(self):
        '''
        Return the 5 year overall survival data.
        '''
        return self._overall_survival

    @property
    def disease_specific_survival(self):
        '''
        Return the 5 year disease specific survival data.
        '''
        return self._disease_specific_survival
