import logging
from pathlib import Path
from typing import Optional, Tuple, Literal
import os

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


def find_neighbors(coor, num_neighbour):
    """
    Find Neighbors of each cell (based on spatial coordinates).
    """
    nbrs = NearestNeighbors(n_neighbors=num_neighbour).fit(coor)
    distances, indices = nbrs.kneighbors(coor, return_distance=True)
    cell_indices = np.arange(coor.shape[0])
    cell1 = np.repeat(cell_indices, indices.shape[1])
    cell2 = indices.flatten()
    distance = distances.flatten()
    spatial_net = pd.DataFrame({"Cell1": cell1, "Cell2": cell2, "Distance": distance})
    return spatial_net


def build_spatial_net(adata, annotation, num_neighbour):
    """
    Build spatial neighbourhood matrix for each spot (cell) based on the spatial coordinates.
    """
    logger.info("------Building spatial graph based on spatial coordinates...")

    coor = adata.obsm["spatial"]
    if annotation is not None:
        logger.info("Cell annotations are provided...")
        spatial_net_list = []
        # Cells with annotations
        for ct in adata.obs[annotation].dropna().unique():
            idx = np.where(adata.obs[annotation] == ct)[0]
            coor_temp = coor[idx, :]
            spatial_net_temp = find_neighbors(coor_temp, min(num_neighbour, coor_temp.shape[0]))
            # Map back to original indices
            spatial_net_temp["Cell1"] = idx[spatial_net_temp["Cell1"].values]
            spatial_net_temp["Cell2"] = idx[spatial_net_temp["Cell2"].values]
            spatial_net_list.append(spatial_net_temp)
            logger.info(f"{ct}: {coor_temp.shape[0]} cells")

        # Cells labeled as nan
        if pd.isnull(adata.obs[annotation]).any():
            idx_nan = np.where(pd.isnull(adata.obs[annotation]))[0]
            logger.info(f"Nan: {len(idx_nan)} cells")
            spatial_net_temp = find_neighbors(coor, num_neighbour)
            spatial_net_temp = spatial_net_temp[spatial_net_temp["Cell1"].isin(idx_nan)]
            spatial_net_list.append(spatial_net_temp)
        spatial_net = pd.concat(spatial_net_list, axis=0)
    else:
        logger.info("Cell annotations are not provided...")
        spatial_net = find_neighbors(coor, num_neighbour)

    return spatial_net.groupby("Cell1")["Cell2"].apply(np.array).to_dict()



def find_neighbors_regional(cell_pos, spatial_net_dict, coor_latent, config, cell_annotations):
    num_neighbour = config.num_neighbour
    annotations = config.annotation

    cell_use_pos = spatial_net_dict.get(cell_pos, [])
    if len(cell_use_pos) == 0:
        return []

    cell_latent = coor_latent[cell_pos, :].reshape(1, -1)
    neighbors_latent = coor_latent[cell_use_pos, :]
    similarity = cosine_similarity(cell_latent, neighbors_latent).reshape(-1)

    if annotations is not None:
        cell_annotation = cell_annotations[cell_pos]
        neighbor_annotations = cell_annotations[cell_use_pos]
        mask = neighbor_annotations == cell_annotation
        if not np.any(mask):
            return []
        similarity = similarity[mask]
        cell_use_pos = cell_use_pos[mask]

    if len(similarity) == 0:
        return []

    indices = np.argsort(-similarity)  # descending order
    top_indices = indices[:num_neighbour]
    cell_select_pos = cell_use_pos[top_indices]
    return cell_select_pos


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
    save_adata: bool = True,
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

    os.makedirs(config.output_file_path, exist_ok=True) # Ensure output directory exists
    np.save(os.path.join(config.output_file_path, "regional_rank.npy"), regional_rank)

    adata.layers[store_layer] = regional_rank
    logger.info(f"Regional rank matrix stored in adata.layers['{store_layer}'] with shape {regional_rank.shape}.")

    if save_adata:
        adata.write(str(path))
        logger.info(f"Modified AnnData saved to {path}")

    return adata, regional_rank
