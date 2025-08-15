import logging

import scanpy as sc
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def _check_use_rep(adata, use_rep):
    if use_rep is None:
        use_rep = 'SEDR'
        logger.info("Use SEDR results for clustering.")

    if use_rep not in adata.obsm and use_rep not in adata.layers:
        raise KeyError(f"use_rep='{use_rep}' not found in adata.obsm or adata.layers.")

    return use_rep


def res_search_fixed_clus_leiden(adata, n_clusters, random_seed, increment=0.01):

    for res in np.arange(0.2, 2, increment):
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['leiden'].unique()) > n_clusters:
            break
    return res-increment


def leiden(adata, n_clusters, random_seed, use_rep, increment=0.01):

    logger.info("Starting Leiden clustering to get {n_clusters} clusters using '{use_rep}' representation...")
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_leiden(adata, n_clusters, increment, random_seed=random_seed)
    sc.tl.leiden(adata, random_state=random_seed, resolution=res)

    return adata


def res_search_fixed_clus_louvain(adata, n_clusters, random_seed, increment=0.01):
    for res in np.arange(0.2, 2, increment):
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['louvain'].unique()) > n_clusters:
            break
    return res-increment

def louvain(adata, n_clusters, random_seed, use_rep, increment=0.01):

    logger.info(f"Starting Louvain clustering to get {n_clusters} clusters using '{use_rep}' representation...")
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_louvain(adata, n_clusters, increment, random_seed=random_seed)
    sc.tl.louvain(adata, random_state=random_seed, resolution=res)


    return adata


def mclust_R(adata, n_clusters, random_seed, use_rep):
    """
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

    adata.obs['mclust'] = pd.Categorical(mclust_res)

    return adata


class ClusteringRunner:
    VALID_METHODS = ['mclust', 'leiden', 'louvain']

    def __init__(self, adata, use_rep=None, random_seed=42):
        self.adata = adata
        use_rep = _check_use_rep(adata, use_rep)
        self.use_rep = use_rep
        self.random_seed = random_seed

    def run(self, method, n_clusters, key_added=None, increment=0.01, plot=False):
        if method not in self.VALID_METHODS:
            raise ValueError(f"Clustering method should be one of {self.VALID_METHODS}, but got '{method}'.")

        if method == 'leiden':
            self.adata = leiden(self.adata, n_clusters, use_rep=self.use_rep,
                                random_seed=self.random_seed, increment=increment)
            col = 'leiden'
        elif method == 'louvain':
            self.adata = louvain(self.adata, n_clusters, use_rep=self.use_rep,
                                 random_seed=self.random_seed, increment=increment)
            col = 'louvain'
        else:  # mclust
            self.adata = mclust_R(self.adata, n_clusters, use_rep=self.use_rep,
                                  random_seed=self.random_seed)
            col = 'mclust'

        target_col = key_added or f"{method}_{self.use_rep}"
        self.adata.obs[target_col] = self.adata.obs[col].astype(int).astype('category')


        if plot:
            sc.pl.spatial(self.adata, color=target_col)

        return target_col

    def save(self, path):
        logger.info(f"Saving AnnData to {path}")
        self.adata.write(path)

