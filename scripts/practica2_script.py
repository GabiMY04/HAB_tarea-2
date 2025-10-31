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
    # Lectura de la red de interracciones que se llama network_dimonds.txt que esta en la carpeta data
    edges = pd.read_csv("data/network_diamond.txt", header=None, names=["Protein1", "Protein2"], sep=",")
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