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
    Leer un .txt y un .tsv y escribir un único fichero de salida tsv.
    """
    import pandas as _pd

    def _read_flexible(path, sep):
        if path is None:
            return _pd.DataFrame()
        if sep is not None:
            return _pd.read_csv(path, sep=sep, dtype=str, engine="python", header=None)
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



def get_gene_ids_online (gene_names, species="human"):
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
    results = []
    seeds = [s for s in seed_genes if s in G]
    disease_module = set(seed_genes)
    if(not seeds):
        print("Ninguna semilla encontrada: convierte la red a ENTREZ/HUGO o usa semillas presentes en la red.")
    else:
        N = G.number_of_nodes()
        module = set(seeds)
        deg = dict(G.degree())


        n_vecinos = {n: 0 for n in G.nodes()}
        for v in module:
            for nbr in G.neighbors(v):
                n_vecinos[nbr] += 1

        candidatos_validos = {n for n in G.nodes() if n_vecinos[n] > 0 and n not in module}

        while len(module) < n_genes and candidatos_validos:
            M = len(module)
            best = None
            best_p = 1.0
            for u in list(candidatos_validos):
                d_u = deg.get(u, 0)
                k_u = n_vecinos.get(u, 0)
                if d_u == 0:
                    pval = 1.0
                else:
                    pval = hypergeom.sf(k_u - 1, N - 1, M, d_u)
                if pval < best_p:
                    best_p = pval
                    best = u
            if best is None:
                candidatos_validos = set()
            else:
                module.add(best)
                results.append((best, best_p))
                candidatos_validos.discard(best)
                for nbr in G.neighbors(best):
                    n_vecinos[nbr] = n_vecinos.get(nbr, 0) + 1
                    if n_vecinos[nbr] == 1 and nbr not in module:
                        candidatos_validos.add(nbr)

    results = pd.DataFrame(results, columns=["Gene", "p-value"])
    return results

# ...existing code...
def hugo_to_entrez_tsv(input_tsv, output_tsv, input_col="Gene", output_col="Entrez",
                       species="human", batch_size=500, drop_unmapped=False):
    """
    Mapear columna input_col (HUGO o mezcla HUGO/Entrez) a Entrez y guardar en output_col.
    - Si un valor de input_col ya es numérico se conserva tal cual.
    - Si drop_unmapped=True se eliminan filas sin mapeo.
    - Devuelve el DataFrame resultante.
    """
    import pandas as _pd
    from mygene import MyGeneInfo

    df = _pd.read_csv(input_tsv, sep="\t", dtype=str, engine="python")
    if input_col not in df.columns:
        input_col = df.columns[0]

    vals = df[input_col].astype(str).fillna("").str.strip()

    # preparar lista de símbolos a mapear (no numéricos, no vacíos)
    to_map = [v for v in vals.unique() if v and not v.isdigit()]
    mg = MyGeneInfo()
    mapping = {}

    for i in range(0, len(to_map), batch_size):
        batch = to_map[i:i+batch_size]
        try:
            res = mg.querymany(batch, scopes="symbol", fields="entrezgene", species=species, as_dataframe=False)
        except Exception:
            for s in batch:
                mapping[s] = None
            continue
        for r in res:
            q = str(r.get("query"))
            if r.get("notfound", False):
                mapping[q] = None
            else:
                eg = r.get("entrezgene")
                mapping[q] = str(eg) if eg is not None else None

    def _map_value(x):
        x = str(x).strip()
        if not x:
            return None
        if x.isdigit():
            return x
        return mapping.get(x, None)

    df[output_col] = vals.map(_map_value)

    if drop_unmapped:
        df = df[df[output_col].notna()].copy()

    df.to_csv(output_tsv, sep="\t", index=False)
    total = len(df)
    mapped = df[output_col].notna().sum()
    print(f"Guardado {output_tsv} — filas: {total}, mapeadas a Entrez: {mapped}")
    return df

############################################
#################### MAIN ##################
############################################


def main():
    import os
    os.makedirs("results", exist_ok=True)

    # Leer aristas (forzar strings)
    df = pd.read_csv("results/merged_output.tsv", header=None, names=["Protein1", "Protein2"],
                     sep="\t", dtype=str, engine="python")

    # Función mínima para mapear una Series (mezcla HUGO/Entrez) a Entrez usando MyGene
    def map_series_to_entrez(series, batch_size=500):
        vals = series.fillna("").astype(str).str.strip()
        is_num = vals.str.match(r'^\d+$', na=False)
        symbols = list(vals[~is_num].unique())
        mg = MyGeneInfo()
        mapping = {}
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            try:
                res = mg.querymany(batch, scopes="symbol", fields="entrezgene", species="human", as_dataframe=False)
            except Exception:
                for s in batch:
                    mapping[s] = ""
                continue
            for r in res:
                q = str(r.get("query"))
                eg = r.get("entrezgene")
                mapping[q] = str(eg) if eg is not None else ""
        def to_entrez(x):
            x = str(x).strip()
            if not x:
                return ""
            if x.isdigit():
                return x
            return mapping.get(x, "")
        return vals.map(to_entrez)

    # Mapear ambas columnas en memoria
    df["Protein1"] = map_series_to_entrez(df["Protein1"])
    df["Protein2"] = map_series_to_entrez(df["Protein2"])

    # Eliminar aristas con algún extremo sin mapeo
    df = df[(df["Protein1"] != "") & (df["Protein2"] != "")].copy()
    df.to_csv("results/merged_output_entrez.tsv", sep="\t", index=False)

    # Crear grafo y ejecutar DIAMOnD
    G = nx.from_pandas_edgelist(df, source="Protein1", target="Protein2")
    print(f"Red cargada con {G.number_of_nodes()} nodos y {G.number_of_edges()} interacciones.")

    gene_ids = get_gene_ids_online(["ENO1", "PGK1", "HK2"])
    print("Mapping:", gene_ids)

    nodes = set(G.nodes())
    seed_genes = []
    for sym, entrez in gene_ids.items():
        if entrez and str(entrez) in nodes:
            seed_genes.append(str(entrez))
        elif sym in nodes:
            seed_genes.append(sym)

    if not seed_genes:
        print("Ninguna semilla encontrada: convierte la red a ENTREZ/HUGO o usa semillas presentes en la red.")
    else:
        print(f"Semillas encontradas: {seed_genes}")
        module_df = algoritmo_diamond(G, seed_genes, n_genes=100)
        module_df.to_csv("results/diamond_module_results.csv", index=False)
        print("Resultados guardados en results/diamond_module_results.csv")
        print(module_df.head())

if __name__ == "__main__":
    main()
