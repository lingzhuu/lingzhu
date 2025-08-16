import logging
import random

import numpy as np
import pandas as pd
from pathlib import Path
import scipy.sparse as sp

import scanpy as sc
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import SEDR

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


def quality_control(adata, params):
    # mode="percentile",          # "percentile" or "absolute"
    # groupby=None,               # per-sample thresholds (e.g., "sample" or "batch")
    # # percentile mode (0–100)
    # low_umi_pct=1,
    # high_umi_pct=99,
    # mt_cap_pct=20,
    # # absolute mode
    # min_umi=None,
    # max_umi=None,
    # mt_cap_abs=None,            # in %
    # # extras
    # drop_mt_genes=True,
    """
    Filter Visium/space spots by UMI, genes, and mt% using either percentiles or absolute thresholds.
    """
    logger.info("Performing quality control...")

    obs = adata.obs
    mt_cap_pct = params.mt_cap_pct

    adata.var['mt'] = adata.var_names.str.startswith(('MT-','mt-'))
    sc.pp.calculate_qc_metrics(adata, layer="count", qc_vars=['mt'], inplace=True)

    def pct_series(name, p):
        if p is None:
            return None
        if params.groupby is None:
            v = np.percentile(obs[name].to_numpy(), p)
            return pd.Series(v, index=obs.index)
        return obs.groupby(params.groupby)[name].transform(lambda x: np.percentile(x.to_numpy(), p))

    if params.mode == "percentile":
        low_umi_s   = pct_series("total_counts",   params.low_umi_pct)
        high_umi_s  = pct_series("total_counts",   params.high_umi_pct)

    elif params.mode == "absolute":
        def abs_series(val): return None if val is None else pd.Series(val, index=obs.index)
        low_umi_s   = abs_series(params.min_umi)
        high_umi_s  = abs_series(params.max_umi)

    else:
        raise ValueError("mode must be 'percentile' or 'absolute'")

    keep = pd.Series(True, index=obs.index)
    if low_umi_s  is not None: keep &= obs["total_counts"]  >= low_umi_s
    if high_umi_s is not None: keep &= obs["total_counts"]  <= high_umi_s
    if mt_cap_pct is not None: keep &= obs["pct_counts_mt"] <= mt_cap_pct
    adata_qc = adata[keep.values].copy()
    logger.info(f"Filtered {adata.shape[0] - adata_qc.shape[0]} cells based on quality control.")
    
    if params.drop_mt_genes:
        # Remove mitochondrial genes
        gene_names = adata_qc.var_names.values.astype(str)
        mt_gene_mask = ~(np.char.startswith(gene_names, "MT-") | np.char.startswith(gene_names, "mt-"))
        adata_qc = adata_qc[:, mt_gene_mask].copy()
        logger.info(f"Removed mitochondrial genes. Remaining genes: {len(gene_names)}.")

    return adata_qc   


def preprocess_data(adata, params):
    """
    Preprocess the Data
    """
    logger.info("Preprocessing data...")

    # HVGs based on count
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=params.feat_cell)

    # Get the pearson residuals
    if params.pearson_residuals:
        sc.experimental.pp.normalize_pearson_residuals(adata, inplace=False)
        pearson_residuals = sc.experimental.pp.normalize_pearson_residuals(
            adata, inplace=False, clip=10
        )
        adata.layers["pearson_residuals"] = pearson_residuals["X"]
    
    # Normalize the data
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
        

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
        self.latent_pca = PCA(n_components=self.params.n_comps, random_state=42).fit_transform(
            self.expression_array
        )
        return self.latent_pca
    
    def run_sedr(self, verbose="whole ST data"):
        # Use PCA if specified
        if self.params.input_pca:
            node_X = self.compute_pca()
        else:
            node_X = self.expression_array

        # Update the input shape
        self.params.n_nodes = node_X.shape[0]
        self.params.feat_cell = node_X.shape[1]

        # Run SEDR
        logger.info(f"Finding latent representations for {verbose}...")
        sedr_net = SEDR.Sedr(node_X, self.graph_dict, mode='clustering')
        sedr_net.train_with_dec()

        del self.graph_dict

        sedr_feat, _, _, _ = sedr_net.process()
        return sedr_feat

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
    set_seed(args.random_seed)

    # Load the ST data
    logger.info(f"Loading ST data of {args.sample_name}...")
    if args.input_format == "visium":
        adata = sc.read_visium(args.input_path)
    else:
        adata = sc.read_h5ad(args.input_path)

    # Normalize 'counts' layer name
    if "counts" in adata.layers and "count" not in adata.layers:
        adata.layers["count"] = adata.layers["counts"]

    def _is_integer_like(x, sample=200000):
        """Check if matrix is approximately integer and non-negative (sampled to avoid full scan)."""
        v = x.data if sp.issparse(x) else np.asarray(x).ravel()
        if v.size == 0:
            return False
        if v.size > sample:
            idx = np.random.choice(v.size, sample, replace=False)
            v = v[idx]
        if v.min() < 0:
            return False
        return np.allclose(v, np.round(v))

    if "count" not in adata.layers:
        if _is_integer_like(adata.X):
            logger.warning("No 'counts' layer found; adata.X appears to contain raw counts and has been copied to adata.layers['counts'].")
            adata.layers["count"] = adata.X.copy()
        else:
            raise ValueError("The data layer should be raw count data.")
    else:
        if not _is_integer_like(adata.layers["count"]):
            raise ValueError("The data layer should be raw count data.")

    # Make variable names unique
    adata.var_names_make_unique()

    logger.info(f"The ST data contains {adata.shape[0]} cells, {adata.shape[1]} genes.")

    
    # Perform quality control
    if args.quality_control:
        adata = quality_control(adata)

    # Preprocess data
    adata = preprocess_data(adata, args)

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


    latent_rep = LatentRepresentationFinder(adata, args)
    latent_gvae = latent_rep.run_gnn_vae(label)
    latent_pca = latent_rep.latent_pca
    latent_sedr = latent_rep.run_sedr()

    # Add latent representations to the AnnData object
    logger.info("Adding latent representations...")
    adata.obsm["latent_GVAE"] = latent_gvae
    adata.obsm["latent_PCA"] = latent_pca
    adata.obsm["latent_SEDR"] = latent_sedr


    # Save the AnnData object
    logger.info("Saving ST data...")
    adata.write(args.hdf5_with_latent_path)