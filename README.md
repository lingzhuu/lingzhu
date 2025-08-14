# GSVA for ST data
### Given domain
find_latent_representation.py

func: qc(optional*), run a given GNN model to obtain latent infomation. (SEDR/HERGAST/PCA)

input: h5ad, visium
output: h5ad w/ latent infomation

### Auto clustering (optional)
clustering_func.py

- leidan
- mcluster (R embedding)
- plot*

input: h5ad w/ latent infomation
output: figure(optional), h5ad w/ latent infomation & clustering info

### Get gene rank list
get_rank.py
