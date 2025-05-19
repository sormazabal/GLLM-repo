from utils.logger import get_logger
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from hdf5storage import savemat, loadmat
from utils.api.tcga_api import (get_metadata_from_project, get_filters_result_from_case, get_filters_result_from_file,
                                download_file, download_files)
from utils.util import check_cache_files
from .tcga_case import TCGA_Case
import torch
import torch.nn as nn
from torch.nn import functional as F

USE_CACHE_ONLY = True
AIC_PREPROCESS = False
AIC_POSTPROCESS = True


class TCGA_Project(object):
    '''
    TCGA Project
    '''
    def __init__(self, project_id, download_directory, cache_directory,
                 well_known_gene_ids=None, genomic_type='tpm', n_threads=1):
        '''
        Initialize the TCGA Project instance with parameters.

        Needed parameters
        :param project_id: Specify the project id.
        :param download_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.

        Optional parameters
        :param well_known_gene_ids: The well-known gene ids of the project.
        :param genomic_type: Specify the wanted genomic type that you want to use.
        :param n_threads: The number of threads to user for concatenating genomic data.
        '''
        self.project_id = project_id

        # Logger
        self.logger = get_logger('preprocess.tcga_project')

        # Directory for download files
        self.download_directory = Path(download_directory)
        self.download_directory.mkdir(exist_ok=True)

        # Directory for cache files
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(exist_ok=True)

        # Maximum number of threads
        self.n_threads = n_threads

        # Get metadatas
        self.project_metadata = self._get_project_metadata(project_id=self.project_id)
        self.case_metadatas = self._get_case_metadatas(project_id=self.project_id)

        # Sorted case_ids
        self.case_ids = sorted(self.case_metadatas)

        # Download files
        self.cases_file_paths = self._download_files(
            project_id=self.project_id,
            case_ids=self.case_ids,
            extract_directory=self.download_directory
        )

        # Data types
        self.genomic_type = genomic_type

        # Create cases
        self.cases = self._create_cases(
            case_ids=self.case_ids,
            directory=self.download_directory,
            case_metadatas=self.case_metadatas,
            cases_file_paths=self.cases_file_paths,
            genomic_type=self.genomic_type
        )

        # Genomic data
        self._genomic = self._concat_genomic_data(
            cases=self.cases,
            cache_directory=self.cache_directory,
            n_threads=self.n_threads
        )

        # Prepare MATLAB data
        self.well_known_gene_ids = well_known_gene_ids
        if self.well_known_gene_ids is not None:
            if AIC_PREPROCESS:
                self.logger.info('AIC Preprocessing')
                self._prepare_matlab_data(
                    genomic_data=self._genomic,
                    well_known_gene_ids=self.well_known_gene_ids,
                    download_directory=self.download_directory,
                    cache_directory=self.cache_directory
                )

            if AIC_POSTPROCESS:
                self.logger.info('AIC Postprocessing')
                self._get_chosen_gene_ids(
                    cache_directory=self.cache_directory,
                    well_known_gene_ids=self.well_known_gene_ids
                )

        # Clinical data
        self._clinical = self._concat_clinical_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )

        # Vital status data
        self._vital_status = self._concat_vital_status_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )

        # Survival data
        self._overall_survival = self._concat_overall_survival_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )
        self._disease_specific_survival = self._concat_disease_specific_survival_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )
        self._survival_time = self._concat_survival_time_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )

        # Primary site data
        self._primary_site = self._concat_primary_site_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )

    def _get_project_metadata(self, project_id):
        '''
        Get the metadata according to project id from TCGA API.

        :param project_id: Specify the project id.
        '''
        if USE_CACHE_ONLY:
            # Look for cached project metadata
            cache_file = self.cache_directory.joinpath(f'project_metadata_{project_id}.json')
            if cache_file.exists():
                self.logger.info(f"Using cached project metadata for {project_id}")
                import json
                with open(cache_file, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"No cached metadata for {project_id}, using mock data")
                return {
                    "id": project_id,
                    "name": f"Mock {project_id}",
                    "summary": {"case_count": 100, "file_count": 500}
                }
    
        # If not using cache only, make the API call
        kwargs = {}
        return get_metadata_from_project(project_id=project_id, **kwargs)

    def _get_case_metadatas(self, project_id):
        '''
        Filter the wanted cases and collect their metadatas from TCGA API.

        :param project_id: Specify the project id.
        '''
        if USE_CACHE_ONLY:
            # Check for cached case metadatas
            case_metadatas_latest_file_path = check_cache_files(
                self.cache_directory.joinpath(project_id), 
                regex=r'case_metadatas_*'
            )
            
            if case_metadatas_latest_file_path:
                self.logger.info(f"Using cached case metadatas for {project_id}")
                try:
                    import pickle
                    with open(case_metadatas_latest_file_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Error loading cached case metadatas: {e}")
                    
            # If we can't load cached data, just use the files we already have
            # by scanning the download directory
            self.logger.info(f"Scanning download directory for existing cases")
            case_metadatas = {}
            cases_dir = self.download_directory.joinpath(project_id)
            if cases_dir.exists():
                for case_dir in cases_dir.glob('*'):
                    if case_dir.is_dir():
                        case_id = case_dir.name
                        case_metadatas[case_id] = {
                            "id": case_id,
                            "submitter_id": case_id,
                            "project": {"project_id": project_id}
                        }
            
            return case_metadatas

    # If not using cache only, make the API call
        case_metadatas = {}

        kwargs = {}
        kwargs['size'] = get_metadata_from_project(project_id, fields=['summary.case_count'])['summary']['case_count']

        # Filter
        kwargs['filters'] = {
            'and': [
                {'=': {'project.project_id': project_id}},
                {'=': {'files.access': 'open'}},
                {'=': {'files.data_type': 'Gene Expression Quantification'}},
                {'=': {'files.experimental_strategy': 'RNA-Seq'}},
                {'=': {'files.data_format': 'TSV'}},
                {'or': [
                    {'=': {'demographic.vital_status': 'Alive'}},
                    {'and': [
                        {'=': {'demographic.vital_status': 'Dead'}},
                        {'not': {'diagnoses.days_to_diagnosis': 'missing'}},
                        {'not': {'demographic.days_to_death': 'missing'}}
                    ]}
                ]}
            ]
        }

        # Expand
        kwargs['expand'] = [
            'annotations',
            'diagnoses',
            'diagnoses.annotations',
            'diagnoses.pathology_details',
            'diagnoses.treatments',
            'demographic',
            'exposures',
            'family_histories',
            'files',
            'follow_ups',
            'follow_ups.molecular_tests',
            'samples',
            'samples.annotations',
            'samples.portions',
            'samples.portions.analytes',
            'samples.portions.analytes.aliquots',
            'samples.portions.analytes.aliquots.annotations',
            'samples.portions.analytes.aliquots.center',
            'samples.portions.analytes.annotations',
            'samples.portions.annotations',
            'samples.portions.center',
            'samples.portions.slides',
            'samples.portions.slides.annotations'
        ]

        for case_metadata in get_filters_result_from_case(**kwargs):
            case_metadatas[case_metadata['id']] = case_metadata

        return case_metadatas

    def _get_case_file_metadatas(self, project_id, case_ids):
        '''
        Filter the wanted files and collect their metadatas from TCGA API.

        :param project_id: Specify the project id.
        :param case_ids: Specify the case ids.
        '''

        if USE_CACHE_ONLY:
            # Instead of making API calls, scan the download directory
            self.logger.info(f"Scanning download directory for existing files")
            cases_file_metadatas = {case_id: {} for case_id in case_ids}
            
            for case_id in case_ids:
                case_dir = self.download_directory.joinpath(case_id)
                if case_dir.exists():
                    for file_path in case_dir.glob('*'):
                        if file_path.is_file():
                            file_name = file_path.name
                            # Use the filename as file_id since we don't have real IDs
                            cases_file_metadatas[case_id][file_name] = file_name
            
            return cases_file_metadatas


        cases_file_metadatas = {case_id: {} for case_id in case_ids}

        kwargs = {}
        kwargs['size'] = get_metadata_from_project(project_id, fields=['summary.file_count'])['summary']['file_count']

        # Filter
        kwargs['filters'] = {
            'and': [
                {'=': {'cases.project.project_id': project_id}},
                {'=': {'access': 'open'}},
                {'or': [
                    {'and': [
                        {'=': {'data_type': 'Clinical Supplement'}},
                        {'=': {'data_format': 'BCR XML'}}
                    ]},
                    {'and': [
                        {'=': {'data_type': 'Gene Expression Quantification'}},
                        {'=': {'experimental_strategy': 'RNA-Seq'}},
                        {'=': {'data_format': 'TSV'}}
                    ]}
                ]}
            ]
        }

        # Fields
        kwargs['fields'] = ['cases.case_id', 'file_name', 'file_id']

        for case_file_metadata in get_filters_result_from_file(**kwargs):
            file_name = case_file_metadata['file_name']
            file_id = case_file_metadata['file_id']

            if len(case_file_metadata['cases']) == 1:
                case_id = case_file_metadata['cases'][0]['case_id']
            else:
                raise ValueError(f'More than one case for {file_name}')

            if case_id in case_ids:
                cases_file_metadatas[case_id][file_name] = file_id
            else:
                continue

        return cases_file_metadatas

    def _download_files(self, project_id, case_ids, extract_directory):
        '''
        Download the related files according to project id and case ids.

        :param project_id: Specify the project id.
        :param case_ids: Specify the case ids.
        :param extract_directory: Specify the directory for downloading and extracting the files.
        '''

        if USE_CACHE_ONLY:
            # Skip downloading, just collect the file paths
            self.logger.info(f"Skipping downloads, using existing files for {project_id}")
            cases_file_paths = {}
            
            for case_id in case_ids:
                file_paths = []
                case_dir = extract_directory.joinpath(case_id)
                
                if case_dir.exists():
                    for file_path in case_dir.glob('*'):
                        if file_path.is_file():
                            file_paths.append(file_path)
                
                if file_paths:
                    cases_file_paths[case_id] = file_paths
            
            return cases_file_paths
        # Get the file metadatas that we wanted
        cases_file_metadatas = self._get_case_file_metadatas(project_id=project_id, case_ids=case_ids)

        total_file_names = []
        for case_id in cases_file_metadatas:
            total_file_names.extend([file_name for file_name in cases_file_metadatas[case_id]])

        # Check if the data exists
        exist_file_names = [file_path.name for file_path in extract_directory.rglob('*') if file_path.is_file()]
        download_file_names = list(set(total_file_names) - set(exist_file_names))
        download_file_ids = []

        for case_id in cases_file_metadatas:
            download_file_ids.extend([
                cases_file_metadatas[case_id][file_name]
                for file_name in cases_file_metadatas[case_id]
                if file_name in download_file_names
            ])

        # Download files from tcga api
        if len(download_file_ids) == 1:
            self.logger.info('Downloading 1 file for {}...'.format(self.project_id))
            download_file(file_id=download_file_ids[0], extract_directory=str(extract_directory))
        elif len(download_file_ids) > 1:
            self.logger.info('Downloading {} files for {}...'.format(len(download_file_ids), self.project_id))
            download_files(file_ids=download_file_ids, extract_directory=str(extract_directory))
        else:
            self.logger.info('All files are downloaded for {}'.format(self.project_id))

        # Seperate each files into the directory, then record the path down
        cases_file_paths = {}
        for case_id in cases_file_metadatas:
            file_paths = []
            for file_name in cases_file_metadatas[case_id]:
                file_path = extract_directory.joinpath(file_name)

                if file_path.exists():
                    new_file_path = extract_directory.joinpath(case_id, file_name)
                    new_file_path.parent.mkdir(exist_ok=True)
                    file_path = file_path.rename(new_file_path)

                file_paths.append(extract_directory.joinpath(case_id, file_name))

            cases_file_paths[case_id] = file_paths

        return cases_file_paths

    def _create_cases(self, case_ids, directory, case_metadatas, cases_file_paths, genomic_type):
        '''
        Create the TCGA Case instance according to case ids and directory.

        :param case_ids: Specify the case ids.
        :param directory: Specify the root directory for the project.
        :param case_metadatas: The metadata of the cases.
        :param case_file_paths: The related file paths of the cases.
        :param genomic_type: Specify the wanted genomic type that you want to use.
        '''
        self.logger.info('Creating {} cases for {}...'.format(len(case_ids), self.project_id))

        cases = {}
        for case_id in case_ids:
            case_params = {}
            case_params['case_id'] = case_id
            case_params['directory'] = directory.joinpath(case_id)
            case_params['case_metadata'] = case_metadatas[case_id]
            case_params['case_file_paths'] = cases_file_paths[case_id]
            case_params['genomic_type'] = genomic_type

            cases[case_id] = TCGA_Case(**case_params)

        return cases

    def _get_genomic_data_wrapper(self, t_case):
        '''
        The multi-threaded wrapper of getting genomic data from a case.

        :param t_case: The dictionary pair of (case_id, TCGA_Case).
        '''
        case_id, case = t_case
        return case_id, case.genomic

    def _concat_genomic_data(self, cases, cache_directory, n_threads):
        '''
        Concatenate the genomic data from the cases.

        :param cases: The TCGA Case instances.
        :param cache_directory: Specify the directory for the cache files.
        :param n_threads: The number of threads to user for concatenating genomic data.

        return: The concatenated genomic data.
        in format: gene_id, case_id1, case_id2, ...
        '''
        # Check if the cache data exists
        genomic_latest_file_path = check_cache_files(cache_directory, regex=f'genomic_{self.genomic_type}_*')
        #print("PAth : ",genomic_latest_file_path)

        if genomic_latest_file_path:
            genomic_latest_file_created_date = genomic_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using genomic {} cache files created at {} for {}'.format(
                self.genomic_type,
                datetime.strptime(genomic_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_genomic_cache = pd.read_csv(genomic_latest_file_path, sep='\t', index_col='gene_id')
            #print(df_genomic_cache)

            return df_genomic_cache

        self.logger.info(f"Concatenating {len(cases)} cases' genomic {self.genomic_type} data for {self.project_id}...")
        df_genomic = pd.DataFrame()

        # Multiple threads
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            for case_id, df_case_genomic in executor.map(self._get_genomic_data_wrapper, cases.items()):
                df_genomic = df_genomic.join(df_case_genomic, how='outer')
        df_genomic.index.rename(name='gene_id', inplace=True)

        # Save the result to cache directory
        df_genomic.to_csv(
            cache_directory.joinpath(f'genomic_{self.genomic_type}_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_genomic

    def _concat_clinical_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the clinical data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
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

        self.logger.info('Concatenating {} cases\' clinical data for {}...'.format(len(cases), self.project_id))
        df_clinical = pd.DataFrame()

        case_ids = genomic_data.columns.to_list()
        for case_id in case_ids:
            df_clinical = df_clinical.join(cases[case_id].clinical, how='outer')

        # Add the name for index
        df_clinical.index.rename(name='clinical', inplace=True)

        # Save the clinical data
        df_clinical.to_csv(
            cache_directory.joinpath(f'clinical_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_clinical

    def _concat_vital_status_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the vital status data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        vital_status_latest_file_path = check_cache_files(cache_directory=cache_directory, regex='vital_status_*')

        if vital_status_latest_file_path:
            vital_status_latest_file_created_date = vital_status_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using vital status cache files created at {} for {}'.format(
                datetime.strptime(vital_status_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_vital_status_cache = pd.read_csv(vital_status_latest_file_path, sep='\t', index_col='vital_status')

            return df_vital_status_cache

        self.logger.info('Concatenating {} cases\' vital status data for {}...'.format(len(cases), self.project_id))
        df_vital_status = pd.DataFrame()

        case_ids = genomic_data.columns.to_list()
        for case_id in case_ids:
            df_vital_status = df_vital_status.join(cases[case_id].vital_status, how='outer')

        # Add the name for index
        df_vital_status.index.rename(name='vital_status', inplace=True)

        # Save the vital status data
        df_vital_status.to_csv(
            cache_directory.joinpath(f'vital_status_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_vital_status

    def _concat_overall_survival_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the 5 years overall survival data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        overall_survival_latest_file_path = check_cache_files(cache_directory, regex=r'overall_survival_*')

        if overall_survival_latest_file_path:
            latest_file_created_date = overall_survival_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using overall survival cache files created at {} for {}'.format(
                datetime.strptime(latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_overall_survival_cache = pd.read_csv(overall_survival_latest_file_path, sep='\t',
                                                    index_col='overall_survival')

            return df_overall_survival_cache

        self.logger.info('Concatenating {} cases\' overall survival data for {}...'.format(len(cases), self.project_id))
        df_overall_survival = pd.DataFrame()

        case_ids = genomic_data.columns.to_list()
        for case_id in case_ids:
            df_overall_survival = df_overall_survival.join(cases[case_id].overall_survival, how='outer')

        # Add the name for index
        df_overall_survival.index.rename(name='overall_survival', inplace=True)

        # Save the overall survival data
        df_overall_survival.to_csv(
            cache_directory.joinpath(f'overall_survival_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_overall_survival

    def _concat_disease_specific_survival_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the 5 years disease specific survival data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        latest_file_path = check_cache_files(cache_directory=cache_directory, regex=r'disease_specific_survival_*')

        if latest_file_path:
            disease_specific_survival_latest_file_created_date = latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using disease specific survival cache files created at {} for {}'.format(
                datetime.strptime(disease_specific_survival_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_disease_specific_survival_cache = pd.read_csv(latest_file_path, sep='\t',
                                                             index_col='disease_specific_survival')

            return df_disease_specific_survival_cache

        self.logger.info(f'Concatenating {len(cases)} cases\' disease specific survival data for {self.project_id}...')
        df_disease_specific_survival = pd.DataFrame()

        case_ids = genomic_data.columns.to_list()
        for case_id in case_ids:
            df_disease_specific_survival = df_disease_specific_survival.join(
                cases[case_id].disease_specific_survival,
                how='outer'
            )

        # Add the name for index
        df_disease_specific_survival.index.rename(name='disease_specific_survival', inplace=True)

        # Save the disease specific survival data
        df_disease_specific_survival.to_csv(
            cache_directory.joinpath(f'disease_specific_survival_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_disease_specific_survival

    def _concat_survival_time_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the survival time data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        survival_time_latest_file_path = check_cache_files(cache_directory=cache_directory, regex=r'survival_time*')

        if survival_time_latest_file_path:
            survival_time_latest_file_created_date = survival_time_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using survival time cache files created at {} for {}'.format(
                datetime.strptime(survival_time_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_survival_time_cache = pd.read_csv(survival_time_latest_file_path, sep='\t', index_col='survival_time')

            return df_survival_time_cache

        self.logger.info('Concatenating {} cases\' survival time data for {}...'.format(len(cases), self.project_id))
        df_survival_time = pd.DataFrame()

        case_ids = genomic_data.columns.to_list()
        for case_id in case_ids:
            df_survival_time = df_survival_time.join(cases[case_id].survival_time, how='outer')

        # Add the name for index
        df_survival_time.index.rename(name='survival_time', inplace=True)

        # Save the survival time data
        df_survival_time.to_csv(
            cache_directory.joinpath(f'survival_time_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_survival_time

    def _concat_primary_site_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the primary site data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        primary_site_latest_file_path = check_cache_files(cache_directory=cache_directory, regex=r'primary_site_*')

        if primary_site_latest_file_path:
            primary_site_latest_file_created_date = primary_site_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using primary site cache files created at {} for {}'.format(
                datetime.strptime(primary_site_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_primary_site_cache = pd.read_csv(primary_site_latest_file_path, sep='\t',
                                                index_col='primary_site', dtype='category')

            return df_primary_site_cache

        self.logger.info('Concatenating {} cases\' primary site data for {}...'.format(len(cases), self.project_id))
        df_primary_site = pd.DataFrame()

        case_ids = genomic_data.columns.to_list()
        for case_id in case_ids:
            df_primary_site = df_primary_site.join(cases[case_id].primary_site, how='outer')

        # Add the name for index
        df_primary_site.index.rename(name='primary_site', inplace=True)

        # Save the primary site data
        df_primary_site.to_csv(
            cache_directory.joinpath(f'primary_site_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_primary_site

    def _distill_biogrid(self, download_directory, biogrid_file_name='BIOGRID-ALL-4.4.203.txt'):
        '''
        Distill the connections from the file download from BioGRID website.

        :param download_directory: Specify the parent directory for the file download from BioGRID website.
        :param biogrid_file_name: Specify the file name that download from BioGRID website.
        '''
        self.logger.info(f'Distilling {biogrid_file_name}...')
        df_biogrid = pd.read_csv(download_directory.joinpath(f'{biogrid_file_name}'), sep='\t', low_memory=False)

        mask = (df_biogrid['Organism Name Interactor A'] == 'Homo sapiens') &\
               (df_biogrid['Organism Name Interactor B'] == 'Homo sapiens')
        columns = ['Official Symbol Interactor A', 'Official Symbol Interactor B']
        df_biogrid = df_biogrid[mask][columns].drop_duplicates().groupby(columns, as_index=False).size().pivot(*columns,
                                                                                                               'size')

        gene_ids = df_biogrid.index.union(df_biogrid.columns)
        df_biogrid = df_biogrid.reindex(index=gene_ids, columns=gene_ids).fillna(0)
        df_biogrid = (df_biogrid + df_biogrid.T) > 0

        df_biogrid.columns.rename(name='', inplace=True)

        return df_biogrid

    # TODO: Try not to save the files
    def _prepare_matlab_data(self, genomic_data, well_known_gene_ids, download_directory, cache_directory):
        '''
        Prepare all the needed files for the AIC code written in MATLAB.

        :param genomic_data: The genomic data.
        :param well_known_gene_ids: The well-known gene ids of the project.
        :param download_directory: Specify the parent directory for the file download from BioGRID website.
        :param cache_directory: Specify the directory for the cache files.
        '''
        gene_ids = genomic_data.index.to_list()
        patient_ids = genomic_data.columns.to_list()

        df_biogrid = self._distill_biogrid(download_directory=download_directory.parent.joinpath('BIOGRID'))

        indices_latest_file_path = check_cache_files(cache_directory=cache_directory, regex=r'indices_*')

        if indices_latest_file_path:
            indices_latest_file_created_date = indices_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using indices cache files created at {} from {} for matlab data'.format(
                datetime.strptime(indices_latest_file_created_date, "%Y%m%d%H%M%S"),
                cache_directory
            ))

            indices_cache = np.load(indices_latest_file_path)
            train_indices = dict(indices_cache)['train']

            genomic_data = genomic_data.iloc[:, train_indices]

            genomic_data.to_csv(cache_directory.joinpath('matlab_genomic.tsv'), sep='\t')
            self.logger.info('Saving genomic data used for matlab data')

        for well_known_gene_id in well_known_gene_ids:
            cache_directory.joinpath(well_known_gene_id).mkdir(exist_ok=True)

            pos_genomic_data, neg_genomic_data, pos_patient_ids, neg_patient_ids = TCGA_Project.stepminer(
                genomic_data=genomic_data.to_numpy(),
                well_known_gene_id=well_known_gene_id,
                gene_ids=gene_ids,
                patient_ids=patient_ids
            )

            anova_pos_genomic_data, anova_neg_genomic_data, anova_gene_ids = TCGA_Project.anova(
                pos_genomic_data=pos_genomic_data,
                neg_genomic_data=neg_genomic_data,
                gene_ids=gene_ids
            )

            df_biogrid_saved = df_biogrid.reindex(index=anova_gene_ids, columns=anova_gene_ids).fillna(False)

            self.logger.info(f'Saving MATLAB data files for {well_known_gene_id}...')
            bind_info = {'bind': df_biogrid_saved.to_numpy(dtype=np.float64)}
            name_pool = {'name_pool': np.asarray([np.asarray(gene) for gene in anova_gene_ids], dtype=object)}
            ANOVA_profile_stemness = {'ANOVA_profile_stemness': anova_pos_genomic_data}
            ANOVA_profile_nonstemness = {'ANOVA_profile_nonstemness': anova_neg_genomic_data}

            savemat(cache_directory.joinpath(well_known_gene_id, f'{well_known_gene_id}_bind_info.mat'),
                    bind_info, format='5')
            savemat(cache_directory.joinpath(well_known_gene_id, f'{well_known_gene_id}_name_pool.mat'),
                    name_pool, format='5')
            savemat(cache_directory.joinpath(well_known_gene_id, f'{well_known_gene_id}_stemness.mat'),
                    ANOVA_profile_stemness, format='5')
            savemat(cache_directory.joinpath(well_known_gene_id, f'{well_known_gene_id}_nonstemness.mat'),
                    ANOVA_profile_nonstemness, format='5')

    def _get_chosen_gene_ids(self, cache_directory, well_known_gene_ids):
        '''
        Calculate marker PRV scores and return the chosen gene ids based on built GINs.

        :param cache_directory: Specify the directory for the cache files.
        :param well_known_gene_ids: The well-known gene ids of the project.
        '''
        gene_candidates = {}

        self.logger.info('Getting chosen gene ids...')

        # Initialize dictionary
        for well_known_gene_id in well_known_gene_ids:
            name_pool_path = cache_directory.joinpath(well_known_gene_id, f'{well_known_gene_id}_name_pool.mat')
            gene_ids = [gene_id.item() for gene_id in loadmat(name_pool_path.as_posix())['name_pool'].squeeze()]

            for gene_id in gene_ids:
                if gene_id not in gene_candidates:
                    gene_candidates[gene_id] = 0

        # Caculate
        for well_known_gene_id in well_known_gene_ids:
            name_pool_path = cache_directory.joinpath(well_known_gene_id, f'{well_known_gene_id}_name_pool.mat')
            stem_path = cache_directory.joinpath(well_known_gene_id, 'Final_reg_ability_stem.mat')
            nonstem_path = cache_directory.joinpath(well_known_gene_id, 'Final_reg_ability_nonstem.mat')

            gene_ids = loadmat(name_pool_path.as_posix())['name_pool'].squeeze()
            stem_weights = loadmat(stem_path.as_posix())['Final_reg_ability_stem']
            nonstem_weights = loadmat(nonstem_path.as_posix())['Final_reg_ability_nonstem']

            # Calculate PRV values
            different_weights = np.abs(stem_weights - nonstem_weights)
            prv_scores = np.zeros(len(different_weights))
            for i in range(len(different_weights)):
                prv_scores[i] = np.sum(different_weights[i, :])

            # Rank genes using PRV scores in descending order
            sort_gene_ids = [gene_id.item() for gene_id in gene_ids[np.argsort(-prv_scores)]]

            for rank_score, gene_id in enumerate(sort_gene_ids):
                if rank_score >= len(gene_ids):
                    rank_score = len(gene_ids) - 1

                gene_candidates[gene_id] = gene_candidates[gene_id] + rank_score + 1

            sub_gene_ids = list(set(gene_candidates.keys()) - set([gene_id.item() for gene_id in gene_ids]))

            for gene_id in sub_gene_ids:
                gene_candidates[gene_id] = gene_candidates[gene_id] + len(gene_ids)

        chosen_gene_ids = set(well_known_gene_ids)

        for chosen_gene_id, _ in sorted(gene_candidates.items(), key=lambda item: item[1]):
            chosen_gene_ids.add(chosen_gene_id)

            if len(chosen_gene_ids) >= 20:
                break

        chosen_gene_ids = [key for _, key in sorted([(gene_candidates[key], key) for key in chosen_gene_ids])]

        self.logger.info(f'Chosen gene ids: {" ".join(chosen_gene_ids)}')

    @staticmethod
    def stepminer(genomic_data, well_known_gene_id, gene_ids, patient_ids, error=0.5):
        '''
        StepMiner with error boundary.

        :param genomic_data: The genomic data.
        :param well_known_gene_ids: The well-known gene ids of the project.
        :param gene_ids: The gene ids.
        :param patient_ids: The patient ids.
        :param error: The error boundary.
        '''
        # Select target gene/marker's expression levels
        well_known_gene_idx = gene_ids.index(well_known_gene_id)
        target_genomic_data = genomic_data[well_known_gene_idx, :]

        target_genomic_data_sorted = np.sort(target_genomic_data)
        num_patients = len(target_genomic_data)

        # Calculate needed statistics
        SSE = np.zeros(num_patients)
        for i in range(num_patients):
            if i == 0:
                low = 0.0
                mean_low = 0.0
            else:
                low = target_genomic_data_sorted[:i]
                mean_low = np.mean(low)

            if i == num_patients - 1:
                high = 0.0
                mean_high = 0.0
            else:
                high = target_genomic_data_sorted[i:]
                mean_high = np.mean(high)

            sse_low = np.sum(np.square(low - mean_low))
            sse_high = np.sum(np.square(high - mean_high))
            SSE[i] = sse_low + sse_high

        # Select cut point and derive the threshold
        cut_point = np.argmin(SSE)
        low = np.mean(target_genomic_data_sorted[:cut_point])
        high = np.mean(target_genomic_data_sorted[cut_point:])
        threshold = (low + high) / 2

        # Remove samples that are close to threshold within +-`error` range and separate all samples to two groups:
        # pos/neg (stemness/nonstemness)
        pos_genomic_data = genomic_data[:, target_genomic_data > (threshold + error)]
        pos_patient_ids = itemgetter(*np.argwhere(target_genomic_data > (threshold + error)).squeeze())(patient_ids)
        neg_genomic_data = genomic_data[:, target_genomic_data < (threshold - error)]
        neg_patient_ids = itemgetter(*np.argwhere(target_genomic_data < (threshold - error)).squeeze())(patient_ids)

        return pos_genomic_data, neg_genomic_data, pos_patient_ids, neg_patient_ids

    @staticmethod
    def anova(pos_genomic_data, neg_genomic_data, gene_ids, p_val=0.05):
        '''
        Use ANOVA to select differentially-expressed genes with threshold of p-value.

        :param pos_genomic_data: The genomic data of the positive group that selected from StepMiner.
        :param neg_genomic_data: The genomic data of the negative group that selected from StepMiner.
        :param gene_ids: The gene ids.
        :param p_val: The threshold of the p-value.
        '''
        num_genes = len(gene_ids)
        p_values = np.zeros(num_genes)
        for i in range(num_genes):
            if pos_genomic_data[i, :].sum() > 0 and neg_genomic_data[i, :].sum() > 0:
                p_values[i] = f_oneway(pos_genomic_data[i, :], neg_genomic_data[i, :])[1]
            else:
                p_values[i] = np.inf

        p_value_idx = np.argwhere(p_values < p_val).squeeze()

        anova_pos_genomic_data = pos_genomic_data[p_values < p_val, :]
        anova_neg_genomic_data = neg_genomic_data[p_values < p_val, :]
        anova_gene_ids = itemgetter(*p_value_idx)(gene_ids)

        return anova_pos_genomic_data, anova_neg_genomic_data, anova_gene_ids

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
    def vital_status(self):
        '''
        Return the vital status data.
        '''
        return self._vital_status

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

    @property
    def survival_time(self):
        '''
        Return the survival time data.
        '''
        return self._survival_time

    @property
    def primary_site(self):
        '''
        Return the primary site data.
        '''
        return self._primary_site
