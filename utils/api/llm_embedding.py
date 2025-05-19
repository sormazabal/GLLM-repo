"""This module provides functions to get node embeddings by genes' names.

Users are required to place the LLM embeddings in a data directory, and specify the path by `data_dir` and `llm`. \
The directory should contain PyTorch tensor (.pt) files, each of them stores the embeddings of a gene, \
and they should be named as: {gene's identifier}.pt.
"""

import os

import torch

from .string_api import _get_identifier
import json


def get_llm_node_embedding(gene_list: list[str], llm: str = 'ESM2_1280', data_dir: str = 'Data/'):
    """Generates node embeddings by genes' names using the specified llm.

    An embeddings of a gene should be stored in: {data_dir}/{llm}/{gene's identifier}.pt.

    Args:
        gene_list (list[str]): list of genes' names.
        llm (str, optional): name of the llm. Defaults to 'ESM2_1280'.
        data_dir (str, optional): directory root that contains the embeddings. Defaults to 'Data/'.
    """
    # make sure the list of genes has only unique genes
    gene_list = list(set(gene_list))
    identifiers_path = os.path.join(data_dir, llm, 'identifiers.json')
    # if the path does not exist, create the identifiers.json file
    if not os.path.exists(identifiers_path):
        identifier_dict = _get_identifier(gene_list)
        with open(identifiers_path, 'w') as f:
            json.dump(identifier_dict, f)
    else:
        with open(identifiers_path, 'r') as f:
            identifier_dict = json.load(f)
    
    nodes_embedding = []
    for gene in gene_list:
        embedding_path = os.path.join(data_dir, llm, f'{identifier_dict[gene]}.pt')
        assert os.path.exists(embedding_path), f'Embedding for {gene} not found at {embedding_path}'
        embedding: torch.Tensor = torch.load(embedding_path)
        

        if llm.startswith('ESM2_1280'): #uses the sequence
            embedding: torch.Tensor = embedding['mean_representations'][33]

        elif llm.startswith('Llama_mean'): #uses the gene description
            embedding: torch.Tensor = embedding.squeeze(0)
            # check if the embedding has inf values
            assert not torch.isinf(embedding).any(), "Infinities in the embedding"

        elif llm.startswith('Llama'): #uses the gene description
            embedding: torch.Tensor = embedding['mean']

        elif llm.startswith('PROTBERT'): #uses the gene description
            embedding: torch.Tensor = embedding['mean']

        elif llm.startswith('Random'): #uses the gene description
            embedding: torch.Tensor = embedding['mean']
            
           

        else:
            embedding: torch.Tensor = embedding.squeeze(0)
            # check if the embedding has inf values
            assert not torch.isinf(embedding).any(), "Infinities in the embedding"


        nodes_embedding.append(embedding)
    return torch.stack(nodes_embedding)


def test():
    BRCA = ['ESR1', 'EFTUD2', 'HSPA8', 'STAU1', 'SHMT2', 'ACTB', 'GSK3B',
            'YWHAB', 'UBXN6', 'PRKRA', 'BTRC', 'DDX23', 'SSR1', 'TUBA1C',
            'SNIP1', 'SRSF5', 'ERBB2', 'MKI67', 'PGR', 'PLAU']
    LUAD = ['HNRNPU', 'STAU1', 'KDM1A', 'SERBP1', 'DHX9', 'EMC1', 'SSR1',
            'PUM1', 'CLTC', 'PRKRA', 'KRR1', 'OCIAD1', 'CDC73', 'SLC2A1',
            'HIF1A', 'PKM', 'CADM1', 'EPCAM', 'ALCAM', 'PTK7']
    COAD = ['HNRNPL', 'HNRNPU', 'HNRNPA1', 'ZBTB2', 'SERBP1', 'RPL4', 'HNRNPK',
            'HNRNPR', 'TFCP2', 'DHX9', 'RNF4', 'PUM1', 'ABCC1', 'CD44',
            'ALCAM', 'ABCG2', 'ALDH1A1', 'ABCB1', 'EPCAM', 'PROM1']
    
    OV = ['TP53', 'CSMD3', 'NF1', 'CDK12', 'FAT3', 'GABRA6', 'RB1', 'MUC1', 
          'MUC16', 'CD9', 'PAX8', 'WT1', 'BRCA1', 'CDKN2A', 'BRCA2', 'FOXL2', 
          'ESR1', 'EFTUD2', 'HSPA8', 'STAU1']

    all_genes = list(set(BRCA + LUAD + COAD))
    #all_genes = list(set( BRCA + LUAD + COAD + OV))
    #print("Total genes: ", len(all_genes))
    #print("Total size of embeddings: ", get_llm_node_embedding(all_genes, llm='Llama-3.1-8B').shape)
    
    #print(get_llm_node_embedding(all_genes, llm='BERT').shape)
    #print(get_llm_node_embedding(all_genes, llm='BioMedicalLlama').shape)
    #print(get_llm_node_embedding(all_genes, llm='ESM2_1280').shape)
    #print(get_llm_node_embedding(all_genes, llm='Llama-3.2-1B').shape)  
    # check if the embeddings have null values
    #print(torch.isnan(get_llm_node_embedding(all_genes, llm='ESM2_1280')).any())

#test()
