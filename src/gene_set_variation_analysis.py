import logging
from pathlib import Path
from typing import Optional, Tuple, Literal
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from tqdm import tqdm, trange

from gsMap.config import LatentToGeneConfigdef 

logger = logging.getLogger(__name__)

def extract_gene_set(rank_list_mat, cluster_information):
    """
    Extract gene sets (sub-dataframes) for each cluster from a full gene-by-spot matrix.

    Parameters
    ----------
    rank_list_mat: 
        A dataframe where rows are spot/cell names and columns are gene names.
        Typically this contains gene expression, ranks, or GSVA scores.
    
    cluster_information:
        A Series where the index matches rank list matrix (spot/cell names), and the values are cluster labels (e.g., from adata.obs['leiden']).
        Each unique label will be used to subset.

    Returns
    -------
    split_rank_list_mats:
        A dictionary where each key is a cluster label (as string), and the value is a df containing rows of that cluster only.
    """
    # Ensure input indices are aligned
    if not rank_list_mat.index.isin(cluster_information.index).all():
        raise ValueError("The indices of rank list must in cluster indices.")

    # Convert cluster labels to strings (for consistent dictionary keys)
    cluster_series = cluster_information.astype(str)

    # Dictionary to hold subsets
    split_rank_list_mats = {}

    for cluster in np.unique(cluster_series):
        spots_in_cluster = cluster_series[cluster_series == cluster].index
        sub_df = rank_list_mat.loc[spots_in_cluster].copy()
        split_rank_list_mats[cluster] = sub_df

    return split_rank_list_mats

    

def gene_set_variation_analysis(config: GeneSetVariantConfig):
    """
    Perform Gene Set Variation Analysis (GSVA) on the latent representations.

    Parameters:
    -----------
    adata: 
        Annotated data matrix with latent representation and clustering infomation.

    Returns:
    --------
    gsva_scores
        GSVA scores for each gene set across all spots.
    """
    

    logger.info("------Loading the spatial data...")
    adata = sc.read_h5ad(config.hdf5_with_latent_path)
    logger.info(f"Loaded spatial data with {adata.n_obs} cells and {adata.n_vars} genes.")

    # Ensure the output directory exists
    if config.output_dir:
        output_file_path = Path(config.output_dir)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract latent representations
    if 'SEDR' not in adata.obsm:
        raise ValueError("Latent representations not found. ")
    
    latent_representations = adata.obsm['SEDR']

    # Extract clustering information
    if 'leiden' not in adata.obs:
        raise ValueError("Clustering information not found.")
    
    cluster_information = adata.obs['leiden'].astype(str)
    
    # Perform GSVA
    logger.info("Starting Gene Set Variation Analysis (GSVA)...")
    sc.tl.gsva(adata, 
               gene_sets=params.gene_sets, 
               method=params.method, 
               min_size=params.min_size, 
               max_size=params.max_size, 
               n_jobs=params.n_jobs)
    
    # Extract GSVA scores
    gsva_scores = adata.obsm['X_gsva']
    
    # Convert to DataFrame for easier handling
    gene_set_names = list(adata.uns['gsva']['gene_sets'].keys())
    gsva_df = pd.DataFrame(gsva_scores, index=adata.obs_names, columns=gene_set_names)
    
    # Save results if output directory is provided
    if output_dir:
        gsva_df.to_csv(output_dir / "gsva_scores.csv")
        logger.info(f"GSVA scores saved to {output_dir / 'gsva_scores.csv'}")
    
    logger.info("GSVA completed successfully.")
    
    return gsva_df
