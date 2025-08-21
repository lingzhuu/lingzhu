import gseapy as gp

def read_txt_file(gene_set_file):
    """
    Prepare gene set database file from GSEA or Enrichr website.

    Parameters
    ----------
    gene_set_file:
        Path to the gene set file.
    """

    if not gene_set_file.lower().endswith(".txt"):
        raise ValueError("Please input a txt file")

    with open(gene_set_file, "r") as f:
        genes = [line.strip() for line in f if line.strip()]

    return genes




class GeneSetParser:
    """
    A wrapper around gseapy.parser functions.
    
    Features:
    - smart_get(): detect local GMT file vs. Enrichr library automatically
    - Wrappers for read_gmt(), get_library(), download_library(), get_library_name(), gsea_cls_parser() and gsea_edb_parser()
    """

    def __init__(self, default_organism="Human"):
        self.default_organism = default_organism

    def get_library_name(self, organism=None):
        org = organism or self.default_organism
        return gp.parser.get_library_name(organism=org)

    def download_library(self, name, organism=None, filename=None):
        org = organism or self.default_organism
        return gp.parser.download_library(name=name, organism=org, filename=filename)

    def get_library(self, name, organism=None, min_size=0, max_size=2000, save=None, gene_list=None):
        org = organism or self.default_organism
        return gp.parser.get_library(
            name=name,
            organism=org,
            min_size=min_size,
            max_size=max_size,
            save=save,
            gene_list=gene_list,
        )

    def read_gmt(self, path):
        return gp.parser.read_gmt(path)
    
    def read_txt(self, path):
        return read_txt_file(path)

    def read_cls(self, cls):
        return gp.parser.gsea_cls_parser(cls)

    def read_edb(self, results_path):
        return gp.parser.gsea_edb_parser(results_path)
    

parser = GeneSetParser()