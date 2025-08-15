import logging
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange

from gsMap.config import LatentToGeneConfig

logger = logging.getLogger(__name__)


def _find_neighbors(coor: np.ndarray, n_neighbors: int) -> pd.DataFrame:
    """KNN based on spatial coordinates."""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coor)
    distances, indices = nbrs.kneighbors(coor, return_distance=True)
    cell_idx = np.arange(coor.shape[0])
    cell1 = np.repeat(cell_idx, indices.shape[1])
    cell2 = indices.ravel()
    distance = distances.ravel()
    return pd.DataFrame({"Cell1": cell1, "Cell2": cell2, "Distance": distance})


def _build_spatial_net(adata, annotation: Optional[str], n_neighbors: int) -> dict[int, np.ndarray]:
    """Build spatial neighbor dictionary; optionally grouped by annotation."""
    logger.info("------Building spatial graph based on spatial coordinates...")
    coor = adata.obsm["spatial"]

    if annotation is not None:
        logger.info("Cell annotations are provided...")
        parts = []

        ann = adata.obs[annotation].values
        for ct in pd.unique(ann[~pd.isnull(ann)]):
            idx = np.where(ann == ct)[0]
            coor_g = coor[idx, :]
            k = min(n_neighbors, coor_g.shape[0])
            df = _find_neighbors(coor_g, k)
            df["Cell1"] = idx[df["Cell1"].to_numpy()]
            df["Cell2"] = idx[df["Cell2"].to_numpy()]
            parts.append(df)
            logger.info(f"{ct}: {coor_g.shape[0]} cells")

        if pd.isnull(ann).any():
            idx_nan = np.where(pd.isnull(ann))[0]
            logger.info(f"NaN: {len(idx_nan)} cells")
            df_all = _find_neighbors(coor, n_neighbors)
            parts.append(df_all[df_all["Cell1"].isin(idx_nan)])

        df_all = pd.concat(parts, axis=0, ignore_index=True)
    else:
        logger.info("Cell annotations are not provided...")
        df_all = _find_neighbors(coor, n_neighbors)

    return df_all.groupby("Cell1")["Cell2"].apply(lambda s: s.to_numpy()).to_dict()


def _regional_topk(
    cell_pos: int,
    spatial_net: dict[int, np.ndarray],
    latent: np.ndarray,
    top_k: int,
    cell_annotations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select top_k neighbors by cosine similarity within spatial neighbors."""
    nbrs = spatial_net.get(cell_pos, np.array([], dtype=int))
    if nbrs.size == 0:
        return nbrs

    q = latent[cell_pos].reshape(1, -1)
    K = latent[nbrs]
    sim = cosine_similarity(q, K).ravel()

    if cell_annotations is not None:
        same = (cell_annotations[nbrs] == cell_annotations[cell_pos])
        if not np.any(same):
            return np.array([], dtype=int)
        nbrs = nbrs[same]
        sim = sim[same]

    if sim.size == 0:
        return np.array([], dtype=int)

    k = min(top_k, sim.size)
    top_idx = np.argpartition(-sim, k - 1)[:k]
    return nbrs[top_idx]


def _maybe_homolog_transform(adata, config: LatentToGeneConfig) -> sc.AnnData:
    """Convert species gene names to human symbols if configured."""
    if config.homolog_file is None or config.species is None:
        return adata

    species_col = f"{config.species}_homolog"
    if species_col in adata.var.columns:
        logger.warning(
            f"Column '{species_col}' already exists; homolog conversion already done. Skipping."
        )
        return adata

    logger.info(f"------Transforming {config.species} genes to HUMAN_GENE_SYM...")
    homologs = pd.read_csv(config.homolog_file, sep="\t", header=0)
    if homologs.shape[1] != 2:
        raise ValueError("Homolog file must have exactly two columns: <species> and HUMAN_GENE_SYM.")

    homologs.columns = [config.species, "HUMAN_GENE_SYM"]
    homologs.set_index(config.species, inplace=True)

    keep = adata.var_names.isin(homologs.index)
    adata = adata[:, keep].copy()
    logger.info(f"{adata.n_vars} genes retained after homolog filtering.")
    if adata.n_vars < 100:
        raise ValueError("Too few genes retained in ST data (<100).")

    adata.var[species_col] = adata.var_names.values

    new_names = homologs.loc[adata.var_names, "HUMAN_GENE_SYM"].values
    adata.var_names = new_names
    adata.var.index.name = "HUMAN_GENE_SYM"
    adata = adata[:, ~adata.var_names.duplicated()].copy()
    logger.info(f"{adata.n_vars} genes retained after removing duplicates.")
    return adata


def get_rank(
    config: LatentToGeneConfig,
    *,
    store_layer: str = "regional_rank",
    agg: Literal["mean", "gmean"] = "mean",
    save: bool = True,
) -> Tuple[sc.AnnData, np.ndarray]:
    """
    Compute regional rank vector for each cell.
    """
    path = Path(config.hdf5_with_latent_path)
    adata = sc.read_h5ad(str(path))

    cell_annotations = None
    if config.annotation is not None:
        logger.info(f"------Cell annotations provided: '{config.annotation}'")
        ann = adata.obs[config.annotation]
        initial = adata.n_obs
        mask = ~pd.isnull(ann)
        if not mask.all():
            adata = adata[mask, :].copy()
            logger.info(f"Removed null annotations: kept {adata.n_obs}/{initial} cells.")
        cell_annotations = adata.obs[config.annotation].values

    adata = _maybe_homolog_transform(adata, config)

    spatial_net = _build_spatial_net(adata, config.annotation, config.num_neighbour_spatial)

    latent_key = config.latent_representation
    if latent_key not in adata.obsm:
        raise KeyError(f"latent_representation '{latent_key}' not found in adata.obsm.")
    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)

    logger.info("------Ranking the spatial data...")
    X = adata.layers["pearson_residuals"] if "pearson_residuals" in adata.layers else adata.X
    n_cells, n_genes = adata.n_obs, adata.n_vars
    ranks = np.zeros((n_cells, n_genes), dtype=np.float32)

    if sp.issparse(X):
        X = X.tocsr()
        for i in tqdm(range(n_cells), desc="Computing ranks per cell"):
            row = X[i].toarray().ravel()
            ranks[i, :] = rankdata(row, method="average")
    else:
        X = np.asarray(X)
        for i in tqdm(range(n_cells), desc="Computing ranks per cell"):
            ranks[i, :] = rankdata(X[i, :], method="average")

    logger.info("------Computing regional rank vectors...")
    top_k = int(config.num_neighbour)
    regional_rank = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in trange(n_cells, desc="Computing regional ranks for each cell"):
        sel = _regional_topk(i, spatial_net, latent, top_k, cell_annotations)
        if sel.size == 0:
            continue
        if agg == "gmean":
            region = ranks[sel, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                vec = np.exp(np.nanmean(np.log(np.maximum(region, 1.0)), axis=0))
            regional_rank[i, :] = vec.astype(np.float32)
        else:
            regional_rank[i, :] = ranks[sel, :].mean(axis=0, dtype=np.float32)

    adata.layers[store_layer] = regional_rank
    logger.info(f"Regional rank matrix stored in adata.layers['{store_layer}'] with shape {regional_rank.shape}.")

    if save:
        adata.write(str(path))
        logger.info(f"Modified AnnData saved to {path}")

    return adata, regional_rank
