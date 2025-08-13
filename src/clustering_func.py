import logging

import scanpy as sc
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def res_search_fixed_clus_leiden(adata, n_clusters, increment=0.01, random_seed=2023):

    for res in np.arange(0.2, 2, increment):
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['leiden'].unique()) > n_clusters:
            break
    return res-increment


def leiden(adata, n_clusters, key_added='SEDR', random_seed=2023):
    sc.pp.neighbor(adata, use_rep=use_rep)
    res = res_search_fixed_clus_leiden(adata, n_clusters, increment=0.01, random_seed=random_seed)
    sc.tl.leiden(adata, random_state=random_seed, resolution=res)

    adata.obs[key_added] = adata.obs['leiden']
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata


def res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=2023):
    for res in np.arange(0.2, 2, increment):
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['louvain'].unique()) > n_clusters:
            break
    return res-increment

def louvain(adata, n_clusters, key_added='SEDR', random_seed=2023):
    sc.pp.neighbor(adata, use_rep=use_rep)
    res = res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=random_seed)
    sc.tl.louvain(adata, random_state=random_seed, resolution=res)

    adata.obs[key_added] = adata.obs['louvain']
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata



def mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR', random_seed=2023):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import os
    os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'
    modelNames = 'EEE'

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_clusters, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata


class ClusteringFunction:
    def __init__(self, adata, args: FindLatentRepresentationsConfig):
        self.params = args
        self.adata = adata

    def clustering(args: FindLatentRepresentationsConfig):
        set_seed(2024)

        # Select highly variable genes
        if args.cluster_method in ["mclust", "leiden", "louvain"]:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.feat_cell)

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
