import math
from datetime import datetime
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
from inmoose.pycombat import pycombat_seq
from torch import from_numpy

from base import BaseDataset
from preprocess import TCGA_Project
from utils.api import get_filters_result_from_project, get_ppi_encoder, get_llm_node_embedding
from utils.logger import get_logger
from utils.util import check_cache_files
from torch_geometric.data import Data as PyG
from dgl import DGLGraph

PROJECT_MAX_COUNT = 1000

def is_detached(tensor):
    return not tensor.requires_grad

def safe_serialize(tensor):
    if not is_detached(tensor):
        tensor = tensor.detach()
    return tensor

# Fix the convert_dgl_to_pyg function
def convert_dgl_to_pyg(dgl_graph):
    """Convert a DGLGraph to a PyG Data object"""
    # Extract node features
    x = dgl_graph.ndata['feat']
    
    # Extract edge indices
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([
        torch.tensor(src, dtype=torch.long), 
        torch.tensor(dst, dtype=torch.long)
    ], dim=0).to(x.device)
    
    # Create PyG Data object
    pyg_graph = PyG(x=x, edge_index=edge_index)
    
    return pyg_graph

def attention_merge(gene_expr, embeddings):
    """
    Merge the gene expression values and LLM embeddings using *attention*
    
    Args:
        gene_expr: Gene expression values [num_nodes, 1]
        embeddings: LLM embeddings [num_nodes, embed_dim]
        
    Returns:
        Merged features using attention mechanism [num_nodes, embed_dim]
    """
    # Flatten out gene expression to calculate attention scores
    # First convert to appropriate shape
    gene_expr_flat = gene_expr.view(-1, 1)  # Ensure shape is [num_nodes, 1]
    
    # Create the attention mechanism
    # 1. Create a learnable parameter for attention (can be fixed for inference!)
    attention_weight = torch.ones(embeddings.shape[1], 1, device=gene_expr.device)
    
    # 2. Calculate attention scores
    attention_scores = torch.matmul(embeddings, attention_weight)  # [num_nodes, 1]
    attention_scores = torch.sigmoid(attention_scores + gene_expr_flat)  # Add gene expression as bias
    
    # 3. Apply attention to combine
    attended_features = embeddings * attention_scores  # Element-wise multiplication with attention
    
    # 4. Add a residual connection with gene expression (for retaining info)
    # Scale gene expression to match embedding dimension
    gene_expr_expanded = gene_expr_flat.expand_as(embeddings)
    merged_features = attended_features + (gene_expr_expanded * 0.1)  # Weighted residual connection
    
    return merged_features



class TCGA_Program_Dataset(BaseDataset):
    '''
    TCGA Program Dataset
    '''
    def __init__(self, project_ids, data_directory, cache_directory, chosen_features=dict(), genomic_type='tpm',
                 target_type='overall_survival', n_threads=1,
                 graph_dataset=False, ppi_score_name='score', ppi_score_threshold=0.0,
                 collect_all_genes=False, batch_effect_correction=False,
                 multi_graph_strategy=None, llm=None, graph_framework= 'dgl', graphbert=False):
        '''
        Initialize the TCGA Program Dataset with parameters.

        Needed parameters
        :param project_id: Specify the project id.
        :param data_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.

        Optional parameters
        :param chosen_features: Specify the features that you want to use.
        :param genomic_type: The genomic type that uses in this project.
        :param target_type: Specify the wanted target type that you want to use.
        :param n_threads: The number of threads to user for concatenating genomic data.
        :param graph_dataset: Whether to use graph or not.
        :param ppi_score_name: The name of the ppi score.
        :param ppi_score_threshold: The threshold of the ppi score.
        '''
        if project_ids not in ['ALL']:
            self.project_ids = project_ids
        else:
            project_filters = {
                '=': {'program.name': 'TCGA'}
            }
            self.project_ids = [
                project_metadata['id'] for project_metadata in get_filters_result_from_project(
                    filters=project_filters,
                    sort='summary.case_count:desc',
                    size=PROJECT_MAX_COUNT
                )
            ]

        # NOTE: If llm is not None, then it must be a graph dataset.
        assert not llm or graph_dataset, 'LLM requires graph dataset'
        self.llm = llm

        if graph_framework == 'dgl':
            self.graph_framework = 'dgl'
            self.graphformer = False
            self.graphbert = False
        elif graph_framework == 'pyg':
            self.graph_framework = 'pyg'
            self.graphformer = True
            self.graphbert = False
        elif graphbert == True:
            self.graph_framework = 'pyg'
            self.graphformer = False
            self.graphbert = True
      
        # TODO: If multi_graph_strategy is not None, then collect_all_genes must be True.
        assert not multi_graph_strategy or collect_all_genes, \
            'Multi graph strategy requires collect_all_genes to be True'
        self.multi_graph_strategy = multi_graph_strategy

        assert not batch_effect_correction or collect_all_genes, \
            'Batch effect correction requires same gene ids for all projects'
        assert not batch_effect_correction or len(self.project_ids) > 1, \
            'Batch effect correction requires more than one project'
        self.batch_effect_correction = batch_effect_correction

        # Logger
        self.logger = get_logger('preprocess.tcga_program_dataset')
        self.logger.info('Creating a TCGA Program Dataset with {} Projects...'.format(len(self.project_ids)))

        # Directory for data and cache files
        self.data_directory = Path(data_directory)
        self.cache_directory = Path(cache_directory)
        

        # Get chosen features
        self.chosen_gene_counts = chosen_features.get('gene_counts', 20)
        self.chosen_project_gene_ids = chosen_features.get('gene_ids', {})
        for project_id in self.project_ids:
            if project_id not in self.chosen_project_gene_ids:
                if project_ids not in ['ALL']:
                    raise ValueError(f'Gene ids is null for {project_id}')
                else:
                    self.chosen_project_gene_ids[project_id] = []
                    self.logger.debug(f'Gene ids is null for {project_id}')

        # TODO: MTL Graph V1: Single large graph for all projects (cancers).
        if collect_all_genes:
            all_gene_ids = list(set().union(*self.chosen_project_gene_ids.values()))
            self.chosen_project_gene_ids = dict.fromkeys(self.project_ids, sorted(all_gene_ids))

        self.chosen_clinical_numerical_ids: list = chosen_features.get('clinical_numerical_ids', [])
        self.chosen_clinical_categorical_ids = chosen_features.get('clinical_categorical_ids', [])
        self.chosen_clinical_ids = self.chosen_clinical_numerical_ids + self.chosen_clinical_categorical_ids

        # Create TCGA_Project instances
        self.tcga_projects: dict[str, TCGA_Project] = {}
        for project_id in self.project_ids:
            self.tcga_project_init_kwargs = {
                'project_id': project_id,
                'download_directory': self.data_directory.joinpath(project_id),
                'cache_directory': self.cache_directory.joinpath(project_id),
                'genomic_type': genomic_type,
                'n_threads': n_threads
            }

            self.tcga_projects[project_id] = TCGA_Project(**self.tcga_project_init_kwargs)

        # Specify the target type
        self.target_type = target_type

        # Specify the genomic type (use graph or not).
        self.graph_dataset = graph_dataset
        self.ppi_score = ppi_score_name
        self.ppi_threshold = ppi_score_threshold

        # Get data from TCGA_Project instance
        self._getdata()

        # Log the information of the dataset.
        self.logger.info('Total {} patients, {} genomic features and {} clinical features'.format(
            len(self.patient_ids), len(self.genomic_ids), len(self.clinical_ids)
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
                self.disease_specific_survivals[self.disease_specific_survivals >= 0]
            ) * 100
        ))
        self.logger.info('{} kinds of primary sites {}'.format(
            len(np.unique(self.primary_sites)), ' / '.join(self.primary_site_ids)
        ))

        # Initialize BaseDataset instance
        self.base_dataset_init_kwargs = {
            'data_root_directory': data_directory,
            'cache_root_directory': cache_directory
        }
        super().__init__(**self.base_dataset_init_kwargs)

    def _getdata(self):

            # Returns:
            # None
        '''
        Get the data from TCGA Program
        '''
        df_genomics = []
        df_clinicals = []
        df_vital_statuses = []
        df_overall_survivals = []
        df_disease_specific_survivals = []
        df_survival_times = []
        df_primary_sites = []
        df_project_ids = []
        train_patient_ids = []
        test_patient_ids = []

        for project_id, tcga_project in self.tcga_projects.items():
            # NOTE: Get the data from TCGA_Project instance.
            df_genomic: pd.DataFrame = tcga_project.genomic.T
            assert self.chosen_project_gene_ids[project_id] != 'ALL', f'No gene ids specified for {project_id}'

                  
            df_genomic = df_genomic[self.chosen_project_gene_ids[project_id]]
            df_clinical: pd.DataFrame = tcga_project.clinical.T[self.chosen_clinical_ids]
            train_indices, test_indices = self._check_cache_train_test_indices(project_id)

            # NOTE: Rename the gene ids to numbers for original multi-task.
            if not self.graph_dataset:
                df_genomic.rename(columns=dict(zip(df_genomic.columns, range(len(df_genomic.columns)))), inplace=True)

            # NOTE: Convert, impute and standardize the clinical data.
            if len(self.chosen_clinical_numerical_ids):
                df_clinical = df_clinical.astype(dict.fromkeys(self.chosen_clinical_numerical_ids, 'float64'))
                if train_indices is not None and test_indices is not None:
                    self.logger.info('Normalize clinical numerical data using training samples only')
                    clinical_mean = df_clinical[self.chosen_clinical_numerical_ids].iloc[train_indices].mean()
                    clinical_std = df_clinical[self.chosen_clinical_numerical_ids].iloc[train_indices].std()
                    train_patient_ids.extend(df_clinical.iloc[train_indices].index.to_list())
                    test_patient_ids.extend(df_clinical.iloc[test_indices].index.to_list())
                else:
                    self.logger.warning('Normalize clinical numerical data using all samples')
                    clinical_mean = df_clinical[self.chosen_clinical_numerical_ids].mean()
                    clinical_std = df_clinical[self.chosen_clinical_numerical_ids].std()
                    train_patient_ids.extend(df_clinical.index.to_list())
                df_clinical = df_clinical.fillna(clinical_mean.to_dict())               # Imputation
                df_clinical[self.chosen_clinical_numerical_ids] -= clinical_mean        # Standardization
                df_clinical[self.chosen_clinical_numerical_ids] /= clinical_std         # Standardization

            # NOTE: One-hot encoding the clinical categorical data.
            if len(self.chosen_clinical_categorical_ids):
                df_clinical = pd.get_dummies(df_clinical, columns=self.chosen_clinical_categorical_ids, dtype=float)
                df_all_tcga_clinical_categorical_ids = self._check_cache_clinical_categorical_ids(project_id)
                df_clinical = df_clinical.reindex(
                    columns=self.chosen_clinical_numerical_ids + df_all_tcga_clinical_categorical_ids.tolist()
                ).fillna(0)

            # NOTE: Append data of the project to the list.
            df_genomics.append(df_genomic)
            df_clinicals.append(df_clinical)
            df_vital_statuses.append(tcga_project.vital_status.T)
            df_overall_survivals.append(tcga_project.overall_survival.T)
            df_disease_specific_survivals.append(tcga_project.disease_specific_survival.T)
            df_survival_times.append(tcga_project.survival_time.T)
            df_primary_site = tcga_project.primary_site.T
            df_project_id = pd.DataFrame(
                data=[self.project_ids.index(project_id)] * len(df_primary_site),
                index=df_primary_site.index,
                columns=['project_id']
            )
            df_primary_sites.append(df_primary_site)
            df_project_ids.append(df_project_id)

        # NOTE: Concatenate the data of all projects.
        df_genomics = pd.concat(df_genomics).fillna(0)                          # Just leave fillna here.
        df_clinicals = pd.concat(df_clinicals).fillna(0)
        df_vital_statuses = pd.concat(df_vital_statuses)
        df_overall_survivals = pd.concat(df_overall_survivals)
        df_disease_specific_survivals = pd.concat(df_disease_specific_survivals)
        df_survival_times = pd.concat(df_survival_times)
        df_primary_sites = pd.concat(df_primary_sites).astype('category')
        df_project_ids = pd.concat(df_project_ids)

        # NOTE: Cache all tcga clinical categorical ids for all projects.
        self._cache_clinical_categorical_ids(df_clinicals)

        # NOTE: Combat correction for gene expression data.
        if self.batch_effect_correction:
            self.logger.info('Combat correction for gene expression data (train samples only)')
            df_genomics.loc[train_patient_ids] = pycombat_seq(
                df_genomics.loc[train_patient_ids].T,
                df_project_ids.loc[train_patient_ids]['project_id'].to_numpy()
            ).T

        df_totals: pd.DataFrame = pd.concat(
            [df_genomics, df_clinicals, df_vital_statuses, df_overall_survivals, df_disease_specific_survivals,
             df_survival_times, df_primary_sites, df_project_ids],
            axis=1
        )

        # NOTE: Transform to graph if graph_dataset is True.
        if self.graph_dataset:
            self._num_nodes = df_genomics.shape[-1]
            self.logger.info(f'Number of nodes for the graph: {self._num_nodes}')
            df_ppis = get_ppi_encoder(
                df_genomics.columns.to_list(),
                score=self.ppi_score,
                threshold=self.ppi_threshold,
                logger=self.logger
            )
            self._genomics = self._process_genomic_as_graph(df_genomics, df_ppis)
        else:
            self._genomics = df_totals[df_genomics.columns].to_numpy(dtype=np.float32)

        self._clinicals = df_totals[df_clinicals.columns].to_numpy(dtype=np.float32)
        self._vital_statuses = df_totals[df_vital_statuses.columns].squeeze().to_numpy(dtype=np.float32)
        self._overall_survivals = df_totals[df_overall_survivals.columns].squeeze().to_numpy(dtype=np.float32)
        self._disease_specific_survivals = df_totals[df_disease_specific_survivals.columns].squeeze().to_numpy(dtype=np.float32)
        self._survival_times = df_totals[df_survival_times.columns].squeeze().to_numpy(dtype=np.float32)
        self._primary_sites = df_totals[df_primary_sites.columns].squeeze().cat.codes.to_numpy()
        self._project_ids = df_totals[df_project_ids.columns].squeeze().to_numpy()
        self._primary_site_ids = tuple(df_totals[df_primary_sites.columns].squeeze().cat.categories.to_list())
        self._patient_ids = tuple(df_totals.index.to_list())
        self._genomic_ids = tuple(df_genomics.columns.to_list())
        self._clinical_ids = tuple(df_clinicals.columns.to_list())

        indices = {
            'train': np.array([i for i, patient_id in enumerate(self._patient_ids) if patient_id in train_patient_ids]),
            'test': np.array([i for i, patient_id in enumerate(self._patient_ids) if patient_id in test_patient_ids])
        }

        for file_path in self.cache_directory.glob('indices_*'):
            if file_path.is_file():
                file_path.unlink()
                self.logger.debug('Removing redundant indices file {}'.format(file_path))

        np.savez(self.cache_directory.joinpath(f'indices_{datetime.now().strftime("%Y%m%d%H%M%S")}.npz'), **indices)
        self.logger.info('Saving train and test indices to {}'.format(self.cache_directory))
        return

    def _process_genomic_as_graph(self, df_genomic: pd.DataFrame, df_ppi: pd.DataFrame):
        """Process genomic data into graph representations.
        
        Args:
            df_genomic: DataFrame containing genomic data with genes as columns
            df_ppi: DataFrame containing protein-protein interactions with src, dst and score columns
            
        Returns:
            list of graph objects (either DGL graphs or PyG graphs depending on configuration)
        """
        # Convert PPI data to torch tensors
        src = from_numpy(df_ppi['src'].to_numpy()).long()
        dst = from_numpy(df_ppi['dst'].to_numpy()).long()
        edge_scores = torch.tensor(df_ppi['score'].to_numpy(), dtype=torch.float32)
        
        # Prepare edge data for different graph formats
        edge_index = torch.stack([src, dst], dim=0)  # Used by PyG
        
        # Get LLM embeddings for genes if enabled
        if self.llm:
            self.logger.info(f"Using {self.llm} embeddings for each gene")
            llm_embeddings = get_llm_node_embedding(df_genomic.columns.to_list(), self.llm, self.data_directory)
            llm_embeddings = safe_serialize(llm_embeddings)
            # Check that embeddings are valid
            assert not torch.isnan(llm_embeddings).any(), 'NaN values in the LLM embeddings'
            assert not torch.isinf(llm_embeddings).any(), 'Infinite values in the LLM embeddings'
        
        # Debug info about graph framework
        self.logger.info(f"Creating graphs with framework: {self.graph_framework}")

        if self.graph_framework == 'dgl':
            src = from_numpy(df_ppi['src'].to_numpy())
            dst = from_numpy(df_ppi['dst'].to_numpy())
            graphs = []
            # Create a graph for each sample (patient).
            for _, row in df_genomic.iterrows():
                g = dgl.graph((src, dst), num_nodes=self._num_nodes)
                g.ndata['feat'] = from_numpy(row.to_numpy()).view(-1, 1).float()

                # LLM embeddings for each gene - USING SCANE
                if self.llm:
                    # Multiply the embedding with gene expression value
                    #g.ndata['feat'] = g.ndata['feat'] * llm_embeddings
                    g.ndata['feat'] = attention_merge(g.ndata['feat'], llm_embeddings)
                    # Check that embeddings are not NaN
                    assert not torch.isnan(g.ndata['feat']).any(), 'NaN values in the embeddings'
                    assert not torch.isinf(g.ndata['feat']).any(), 'Infinite values in the embeddings'
                    
                g = dgl.add_reverse_edges(g)
                graphs.append(g)

            # Fix the assertion syntax error - separate it from the comment
            assert isinstance(graphs[0], dgl.DGLGraph), f"Expected DGLGraph but got {type(graphs[0])}"
            self.logger.info(f"Graph format: {type(graphs[0])}")
            return graphs

        else:
            graphs = []
            for _, row in df_genomic.iterrows():
                # Prepare gene expression data as node features
                gene_expression = torch.tensor(row.to_numpy(), dtype=torch.float32)
                node_features = gene_expression.unsqueeze(-1) * llm_embeddings if self.llm else gene_expression.unsqueeze(-1)
                
                # Create PyG graph
                if self.graphbert:
                    # Add positional encoding for GraphBERT
                    positional_encoding = torch.arange(len(node_features)).unsqueeze(-1).float()
                    g = PyG(x=node_features, edge_index=edge_index, pos=positional_encoding)
                else:
                    # Create PyG graph without positional encoding for Graphormer
                    g = PyG(x=node_features, edge_index=edge_index, edge_attr=edge_scores)
                            
                graphs.append(g)              
                           
            assert isinstance(graphs[0], PyG), f"Expected PyG but got {type(graphs[0])}"
            return graphs

    def _check_cache_train_test_indices(self, project_id: str):
        indices_latest_file_path = check_cache_files(self.cache_directory.joinpath(project_id), r'indices_*')
        if indices_latest_file_path:
            indices_latest_file_created_date = indices_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using indices cache files created at {} from {}'.format(
                datetime.strptime(indices_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.cache_directory.joinpath(project_id)
            ))
            indices_cache = np.load(indices_latest_file_path)
            return indices_cache['train'], indices_cache['test']
        return None, None

    def _check_cache_clinical_categorical_ids(self, project_id: str):
        lts_file = check_cache_files(self.cache_directory.joinpath(project_id), r'all_tcga_clinical_categorical_ids_*')
        assert lts_file is not None, 'No all tcga clinical categorical ids cache files found'
        created_date = lts_file.name.split('.')[0].split('_')[-1]
        self.logger.info('Using all tcga clinical categorical ids cache files created at {} for {}'.format(
            datetime.strptime(created_date, "%Y%m%d%H%M%S"),
            project_id
        ))
        clinical_categorical_ids: pd.Series = pd.read_csv(lts_file, sep='\t', header=None).squeeze()
        return clinical_categorical_ids

    def _cache_clinical_categorical_ids(self, df_clinicals: pd.DataFrame):
        filtered_ids = get_filters_result_from_project(filters={'=': {'program.name': 'TCGA'}}, size=PROJECT_MAX_COUNT)
        #print("filtered_ids:", filtered_ids)
        if len(self.project_ids) == len(filtered_ids):
            all_tcga_clinical_categorical_columns = df_clinicals.columns[
                ~df_clinicals.columns.isin(self.chosen_clinical_numerical_ids)
            ].to_series()
            for project_id in self.project_ids:
                all_cached_clinical_categorical_ids = self._check_cache_clinical_categorical_ids(project_id)
                if set(all_tcga_clinical_categorical_columns) == set(all_cached_clinical_categorical_ids):
                    continue
                all_tcga_clinical_categorical_columns.to_csv(self.cache_directory.joinpath(
                    project_id,
                    f'all_tcga_clinical_categorical_ids_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'
                ), sep='\t', index=False, header=False)
                self.logger.info('Saving all tcga clinical categorical ids to {}'.format(
                    self.cache_directory.joinpath(project_id)
                ))
            return
        self.logger.warning('Skip saving all tcga clinical categorical ids')

    def __getitem__(self, index):
        '''
        Support the indexing of the dataset
        '''
        if self.graph_dataset:
            graph = self._genomics[index]
            
            # Ensure we're passing the graph in the right format for the model
            # If using PyG (graphformer), return the graph as is
            if self.graphformer:
                # For PyG, the graph is already in the right format
                pass
            else:
                # For DGL, ensure it's not a list (GCN FE expects a single graph)
                if isinstance(graph, list):
                    # If it's a list, batch the graphs
                    graph = dgl.batch(graph)
                
            return (graph, self._clinicals[index], index, self._project_ids[index]), \
                   (self.targets[index], self._survival_times[index], self._vital_statuses[index])
        else:
            return (self._genomics[index], self._clinicals[index], index, self._project_ids[index]), \
                   (self.targets[index], self._survival_times[index], self._vital_statuses[index])

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
        if not self.graph_dataset:
            return np.hstack((self._genomics, self._clinicals))
        
        # Extract node features from graphs
        genomic_data = []
        for g in self._genomics:
            if hasattr(g, 'x'):
                # PyG graph
                features = g.x.view(1, -1).cpu().detach().numpy()
            else:
                # DGL graph
                features = g.ndata['feat'].view(1, -1).cpu().detach().numpy()
            genomic_data.append(features)
        
        genomic_data = np.array(genomic_data)
        genomic_data = np.squeeze(genomic_data, axis=1)
        
        return np.hstack((genomic_data, self._clinicals))

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
    def vital_statuses(self):
        '''
        Return the vital status data.
        '''
        return self._vital_statuses

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
    def survival_times(self):
        '''
        Return the survival time data.
        '''
        return self._survival_times

    @property
    def primary_sites(self):
        '''
        Return the primary site data.
        '''
        return self._primary_sites

    @property
    def primary_site_ids(self):
        '''
        Return the primary site ids.
        '''
        return self._primary_site_ids

    @property
    def weights(self):
        '''
        Return the weights for each data.
        '''
        weights = np.zeros_like(self._project_ids, dtype='float64')
        for i in range(self._project_ids.min(), self._project_ids.max() + 1):
            weights[self._project_ids == i] = math.sqrt(1.0 / (self._project_ids == i).sum())

        # Label (target) balancing.
        # weights_labels = np.zeros_like(weights, dtype='float64')
        # for i in range(self.targets.min().astype(int), self.targets.max().astype(int) + 1):
        #     weights_labels[self.targets != i] = np.sqrt(1.0 / (self.targets == i).sum())
        #     weights_labels[self.targets != i] = np.sqrt((self.targets == i).sum())

        # Task and label balancing.
        # for task in range(self._project_ids.min(), self._project_ids.max() + 1):
        #     for label in range(self.targets.min().astype(int), self.targets.max().astype(int) + 1):
        #         weights[(self._project_ids == task) & (self.targets == label)] = math.sqrt(
        #             1.0 / ((self._project_ids == task) & (self.targets == label)).sum()
        #         )
        return weights

    @property
    def targets(self):
        '''
        Return the target data according to target_type.
        '''
        if self.target_type == 'overall_survival':
            return self._overall_survivals
        elif self.target_type == 'disease_specific_survival':
            return self._disease_specific_survivals
        elif self.target_type == 'primary_site':
            return self._primary_sites
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
