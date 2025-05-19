import json
import random
from collections import OrderedDict
from pathlib import Path

import dgl
import numpy
import torch
import yaml
from six import iteritems


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def read_yaml(fname):
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    yaml.add_constructor(_mapping_tag, dict_constructor)

    fname = Path(fname)

    with fname.open('rt') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def write_yaml(content, fname):
    def dict_representer(dumper, data):
        return dumper.represent_dict(iteritems(data))

    yaml.add_representer(OrderedDict, dict_representer)

    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle)


def set_random_seed(seed: int):
    # Set python seed
    random.seed(seed)

    # Set numpy seed
    numpy.random.seed(seed)

    # Set pytorch seed, disable benchmarking and avoiding nondeterministic algorithms
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')
  

    # Set DGL seed.
    dgl.seed(seed)


def check_cache_files(cache_directory, regex):
    '''
    Check if the cache file exists.

    :param cache_directory: Specify the directory for the cache files.
    :param regex: Specify the patterns for searching in cache_directory.
    '''
    cache_file_paths = [file_path for file_path in cache_directory.rglob(regex) if file_path.is_file()]

    latest_file_path = None
    for cache_file_path in cache_file_paths:
        cache_file_name = cache_file_path.name.split('.')[0]

        if not latest_file_path:
            latest_file_path = cache_file_path
        else:
            latest_file_name = latest_file_path.name.split('.')[0]

            if int(cache_file_name.split('_')[-1]) > int(latest_file_name.split('_')[-1]):
                latest_file_path = cache_file_path

    return latest_file_path


def config_add_subdict_key(config: dict = None, prefix: str = '', sep: str = '.'):
    """Add the key of the sub-dict to the parent dict recursively with the separator."""
    if config is None:
        return None
    flatten_dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flatten_dict.update(config_add_subdict_key(prefix=f'{prefix}{key}{sep}', sep=sep, config=value))
        flatten_dict[f'{prefix}{key}'] = value
    return flatten_dict


def override_n_genes(config: dict):
    if isinstance(config, dict):
        all_listed_genes = config['datasets']['chosen_features']['gene_ids']
        if isinstance(all_listed_genes, list):
            n_genes = len(all_listed_genes)
        elif isinstance(all_listed_genes, dict):
            genes_set = set()
            for listed_genes in all_listed_genes.values():
                genes_set.update(listed_genes)
            n_genes = len(genes_set)
        else:
            raise ValueError(f'Unknown type of chosen_features: {type(all_listed_genes)}')
        for model_name in config['models'].keys():
            if 'n_genes' in config['models'][model_name]:
                config['models'][model_name]['n_genes'] = n_genes
        return

    genes = config['datasets.TCGA_BLC.args.chosen_features.gene_ids'] if 'TCGA_BLC' in config['datasets'] else None
    try:
        if genes is None:   # TCGA_Project_Dataset.
            n_genes = config['models.Feature_Extractor.args.n_genes']
        elif isinstance(genes, list):
            n_genes = len(genes)
        elif isinstance(genes, dict):
            all_selected_genes = set()
            for genes in genes.values():
                all_selected_genes.update(genes)
            n_genes = len(all_selected_genes)
        config['models']['Feature_Extractor']['args']['n_genes'] = n_genes
    except KeyError:        # No Feature_Extractor in config file or no n_genes in Feature_Extractor args.
        pass
