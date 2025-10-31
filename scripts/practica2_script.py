############################################
############## LIBRERIAS ###################
############################################
import networkx as nx
import pandas as pd
from scipy.stats import hypergeom
from mygene import MyGeneInfo


############################################
######### FUNCIONES ########################
############################################

def merge_txt_and_tsv(txt_path, tsv_path, out_path, txt_sep=None, tsv_sep="\t", axis=0, index=False):
    """
    Leer un .txt y un .tsv y escribir un único fichero de salida tsv
    """
    import pandas as _pd

    def _read_flexible(path, sep):
        if path is None:
            return _pd.DataFrame()
        if sep is not None:
            return _pd.read_csv(path, sep=sep, dtype=str, engine="python", header=None)
        # intentar detectar separador muestreando las primeras líneas
        sample_text = ""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for _ in range(20):
                    line = fh.readline()
                    if not line:
                        break
                    sample_text += line
        except Exception:
            sample_text = ""

        for s in ["\t", ",", ";", "|"]:
            if s in sample_text:
                return _pd.read_csv(path, sep=s, dtype=str, engine="python", header=None)

        # fallback: leer línea a línea como una sola columna (evita sep="\n")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = [ln.rstrip("\n\r") for ln in fh if ln.strip() != ""]
        except Exception:
            lines = []
        return _pd.DataFrame(lines, dtype=str)

    df_txt = _read_flexible(txt_path, txt_sep)
    df_tsv = _read_flexible(tsv_path, tsv_sep)

    if df_txt.empty:
        result = df_tsv
    elif df_tsv.empty:
        result = df_txt
    else:
        if axis == 0:
            result = _pd.concat([df_txt, df_tsv], axis=0, ignore_index=True, sort=False)
        else:
            result = _pd.concat([df_txt.reset_index(drop=True), df_tsv.reset_index(drop=True)], axis=1)

    result.to_csv(out_path, sep="\t", index=index, header=False)
    return out_path



def get_gene_ids_online(gene_names, species="human"):
    """
    Se pasan los nombres que hay que buscar en la red y se devuelven los IDs
    """
    mg = MyGeneInfo()
    res = mg.querymany(gene_names, scopes="symbol", fields="entrezgene", species=species)
    # Construir un diccionario símbolo -> entrez (como string)
    mapping = {}
    for r in res:
        sym = r.get('query')
        if 'entrezgene' in r and r.get('entrezgene') is not None:
            mapping[sym] = str(r.get('entrezgene'))
    return mapping

def algoritmo_diamond(G, seed_genes, n_genes=200):
    """
    Es el algoritmo diamond que recibe unos ids de genes "seed_genes" y una 
    red G y devuelve un dataframe con los n_genes mas significativos
    """
    disease_module = set(seed_genes)
    all_genes = set(G.nodes())
    total_edges = G.number_of_edges()

    results = []

    while len(disease_module) < n_genes:
        candidate_scores = {}

        for gene in all_genes - disease_module:
            # conexiones del gen con el módulo actual
            k = G.degree(gene)
            K = sum(G.degree(n) for n in disease_module)
            x = sum(1 for n in G.neighbors(gene) if n in disease_module)

            # test hipergeométrico
            M = 2 * total_edges
            n = K
            pval = hypergeom.sf(x - 1, M, n, k)

            candidate_scores[gene] = pval

        best_gene = min(candidate_scores, key=candidate_scores.get)
        best_pval = candidate_scores[best_gene]

        disease_module.add(best_gene)
        results.append((best_gene, best_pval))

        if len(results) % 10 == 0:
            print(f"{len(results)} genes añadidos...")

    return pd.DataFrame(results, columns=["Gene", "p-value"])

############################################
#################### MAIN ##################
############################################

def main():
    merge_txt_and_tsv("data/network_dimonds.txt", "data/string_network_filtered_hugo-400.tsv", "results/merged_output.tsv")
    edges = pd.read_csv("results/merged_output.tsv", header=None, names=["Protein1", "Protein2"], sep="\t")
    # Creamos la red mapeando el archivo leido
    G = nx.from_pandas_edgelist(edges, source="Protein1", target="Protein2")
    print(f"Red cargada con {G.number_of_nodes()} nodos y {G.number_of_edges()} interacciones.")
    gene_ids = get_gene_ids_online(["ENO1", "PGK1", "HK2"])
    print(gene_ids)

    seed_genes = gene_ids
    # Comprobamos que existan en la red
    seed_genes = [g for g in seed_genes if g in G.nodes()]
    print(f"Semillas válidas encontradas en la red: {len(seed_genes)}")



    module_df = algoritmo_diamond(G, seed_genes, n_genes=100)
    module_df.to_csv("results/diamond_module_results.csv", index=False)
    print("Resultados guardados en diamond_module_results.csv")
    print(module_df.head())

if __name__ == "__main__":
    main()