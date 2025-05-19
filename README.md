# GLLM: Multi-Task Graph LLM for Cancer Prognosis Prediction

This repository contains the implementation of GLLM, a novel framework that integrates Large Language Model (LLM) embeddings with Graph Neural Networks (GNNs) to improve cancer survival prediction using genomic data.

## Overview

Cancer prognosis prediction is a challenging task due to the complexity of gene interactions and limited patient data. GLLM addresses these constraints by combining the representational power of LLMs with the relational modeling capabilities of GNNs in a multi-task learning framework.

Our approach:
- Analyzes samples across different cancer types with similar aggression levels
- Leverages shared biological mechanisms for improved predictive power
- Systematically evaluates various LLMs (sequence-based and description-based models)

## Repository Structure

The repository is organized as follows:

- `/dataset`: Code for handling TCGA datasets
- `/datasets_manager`: Classes for managing multiple datasets
- `/model`: Model architectures for feature extraction and classification
- `lit_models.py`: PyTorch Lightning model implementations
- `/utils`: Utility functions for data processing, logging, etc.
- `/config`: YAML configuration files for different experiments
- `/scripts`: Scripts for data preprocessing and analysis


## Models

Our repository implements several models:
1. Non-graph MTL: A baseline multi-task learning model
2. GNN: A graph-based multi-task learning model using GCN
3. GLLM (our proposed model): Integrates LLM embeddings with GNNs

## Requirements

- PyTorch
- PyTorch Lightning
- DGL (Deep Graph Library)
- pandas
- numpy
- scikit-learn
- tqdm
- YAML
- PyG (for the benchmarks)

## Usage

### Training

To train a model, use the `light.py` script with a configuration file:

```bash
python light.py -c config/light/your_config.yaml

