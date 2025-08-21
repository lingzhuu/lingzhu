import logging
from pathlib import Path
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import genesets

from gsMap.config import LatentToGeneConfigdef 

logger = logging.getLogger(__name__)
    

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


def plot_heatmap(gsva_df, output_file_path):
    """
    Plot a heatmap of GSVA scores.

    Parameters:
    -----------
    gsva_df: 
        DataFrame containing GSVA scores.
    
    output_file_path: 
        Optional path to save the heatmap image.
    """


    plt.figure(figsize=(10, 8))
    sns.heatmap(gsva_df, cmap='viridis', annot=False)
    
    if output_file_path:
        plt.savefig(output_file_path)
        logger.info(f"Heatmap saved to {output_file_path}")
    
    plt.show()