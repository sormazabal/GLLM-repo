import argparse
from concurrent.futures import ThreadPoolExecutor
from utils.api import get_filters_result_from_project
from dataset import TCGA_Project_Dataset, TCGA_Program_Dataset
from utils.logger import setup_logging
from pathlib import Path
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)

PROJECT_MAX_COUNT = 1000
all_tcga_project_filters = {
    '=': {'program.name': 'TCGA'}
}

DOWNLOAD_ROOT_DIRECTORY = './Data'
CACHE_ROOT_DIRECTORY = './Cache'
GENOME_TYPE = 'tpm'
N_THREADS = 16

CHOSEN_FEATURES = {
    'clinical_numerical_ids': ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth'],
    'clinical_categorical_ids': ['gender', 'race', 'ethnicity']
}

def create_project_dataset(project):
    project_id = project['id']

    TCGA_Project_Dataset(
        project_id=project_id,
        chosen_features=CHOSEN_FEATURES,
        data_directory='/'.join([DOWNLOAD_ROOT_DIRECTORY, project_id]),
        cache_directory='/'.join([CACHE_ROOT_DIRECTORY, project_id]),
        genomic_type=GENOME_TYPE,
        n_threads=N_THREADS
    )

def main(args):
    if args.type == 'program':
        TCGA_Program_Dataset(
            project_ids='ALL',
            chosen_features=CHOSEN_FEATURES,
            data_directory=DOWNLOAD_ROOT_DIRECTORY,
            cache_directory=CACHE_ROOT_DIRECTORY,
            genomic_type=GENOME_TYPE,
            n_threads=N_THREADS
        )

        # TODO Need to remove indices file that created from TCGA Program Dataset
    elif args.type == 'project':
        with ThreadPoolExecutor() as executor:
            executor.map(
                create_project_dataset,
                get_filters_result_from_project(
                    filters=all_tcga_project_filters,
                    sort='summary.case_count:desc',
                    size=PROJECT_MAX_COUNT
                )
            )

if __name__ == '__main__':
    log_dir = Path('./Logs/Get_All_TCGA/')
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path('./Logs/Get_All_TCGA/'))

    args = argparse.ArgumentParser(description='Get All TCGA')
    args.add_argument('-t', '--type', default='program', type=str,
                      help='type of dataset (default: program)')

    args = args.parse_args()
    main(args)
