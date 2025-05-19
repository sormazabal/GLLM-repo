import argparse
from dataset import TCGA_Project_Dataset
import dataset as module_dataset
import datasets_manager as module_datasets_manager
from parse_config import ConfigParser
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)

def main(config):
    Datasets = {project_id: config.init_obj(
            f'datasets.{project_id}',
            module_dataset
        )
        for project_id in config['datasets']
    }

    Datasets_Manager = getattr(
        module_datasets_manager,
        config['datasets_manager.type']
    )(datasets=Datasets, config=config)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Prepare MATLAB')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    config = ConfigParser.from_args(args)
    main(config)
