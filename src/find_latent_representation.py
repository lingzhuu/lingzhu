import logging
import random

import numpy as np
import pandas as pd
from pathlib import Path

import scanpy as sc
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from gsMap.config import FindLatentRepresentationsConfig
from gsMap.GNN.adjacency_matrix import construct_adjacency_matrix
from gsMap.GNN.train import ModelTrainer


logger = logging.getLogger(__name__)


def set_seed(seed_value):
    """
    Set seed for reproducibility in PyTorch and other libraries.
    """
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        logger.info("Using GPU for computations.")
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    else:
        logger.info("Using CPU for computations.")


def preprocess_data(adata, params):
    """
    Preprocess the Data
    """
    logger.info("Preprocessing data...")

    if params.data_layer in adata.layers.keys():
        logger.info(f"Using data layer: {params.data_layer}...")
        adata.X = adata.layers[params.data_layer].copy()
    elif params.data_layer == "X":
        logger.info(f"Using data layer: {params.data_layer}...")
        if adata.X.dtype == "float32" or adata.X.dtype == "float64":
            logger.warning("The data layer should be raw count data")
    else:
        raise ValueError(f"Invalid data layer: {params.data_layer}, please check the input data.")

    if params.data_layer in ["count", "counts", "X"]:

        # Normalize the data
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Get the pearson residuals
        if params.pearson_residuals:
            sc.experimental.pp.normalize_pearson_residuals(adata, inplace=False)
            pearson_residuals = sc.experimental.pp.normalize_pearson_residuals(
                adata, inplace=False, clip=10
            )
            adata.layers["pearson_residuals"] = pearson_residuals["X"]


    return adata


class LatentRepresentationFinder:
    def __init__(self, adata, args: FindLatentRepresentationsConfig):
        self.params = args

        if "pearson_residuals" in adata.layers:
            self.expression_array = (
                adata[:, adata.var.highly_variable].layers["pearson_residuals"].copy()
            )
        else:
            self.expression_array = adata[:, adata.var.highly_variable].X.copy()
            self.expression_array = sc.pp.scale(self.expression_array, max_value=10)

        # Construct the neighboring graph
        self.graph_dict = construct_adjacency_matrix(adata, self.params)

    def compute_pca(self):
        self.latent_pca = PCA(n_components=self.params.n_comps).fit_transform(
            self.expression_array
        )
        return self.latent_pca

    def run_gnn_vae(self, label, verbose="whole ST data"):
        # Use PCA if specified
        if self.params.input_pca:
            node_X = self.compute_pca()
        else:
            node_X = self.expression_array

        # Update the input shape
        self.params.n_nodes = node_X.shape[0]
        self.params.feat_cell = node_X.shape[1]

        # Run GNN
        logger.info(f"Finding latent representations for {verbose}...")
        gvae = ModelTrainer(node_X, self.graph_dict, self.params, label)
        gvae.run_train()

        del self.graph_dict

        return gvae.get_latent()


def run_find_latent_representation(args: FindLatentRepresentationsConfig):
    set_seed(2024)

    # Load the ST data
    logger.info(f"Loading ST data of {args.sample_name}...")
    if args.input_format == "visium":
        adata = sc.read_visium(args.input_path)
    else:
        adata = sc.read_h5ad(args.input_path)

    if "count" not in adata.layers:
        adata.layers["count"] = adata.X.copy()

    adata.var_names_make_unique()
    # Filter genes with low expression
    sc.pp.filter_genes(adata, min_cells=args.min_cells)

    logger.info(f"The ST data contains {adata.shape[0]} cells, {adata.shape[1]} genes.")

    # Load the cell type annotation
    if args.annotation is not None:
        # Remove cells without enough annotations
        adata = adata[~adata.obs[args.annotation].isnull()]
        num = adata.obs[args.annotation].value_counts()
        valid_annotations = num[num >= 30].index.to_list()
        adata = adata[adata.obs[args.annotation].isin(valid_annotations)]

        le = LabelEncoder()
        label = le.fit_transform(adata.obs[args.annotation])
    else:
        label = None

    # Preprocess data
    adata = preprocess_data(adata, args)

    latent_rep = LatentRepresentationFinder(adata, args)
    latent_gvae = latent_rep.run_gnn_vae(label)
    latent_pca = latent_rep.latent_pca

    # Add latent representations to the AnnData object
    logger.info("Adding latent representations...")
    adata.obsm["latent_GVAE"] = latent_gvae
    adata.obsm["latent_PCA"] = latent_pca


    # Save the AnnData object
    logger.info("Saving ST data...")
    adata.write(args.hdf5_with_latent_path)