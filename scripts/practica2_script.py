############################################
############## LIBRERIAS ###################
############################################
import argparse
import sys
import networkx as nx
import pandas as pd
from scipy.stats import hypergeom
from mygene import MyGeneInfo
import os
############################################
######### FUNCIONES ########################
############################################

def merge_txt_and_tsv(txt_path, tsv_path, out_path, txt_sep=None, tsv_sep="\t", axis=0, index=False):
    """
    Leer un .txt y un .tsv y escribir un único fichero de salida tsv.
    """

    def _read_flexible(path, sep):
        if path is None:
            return pd.DataFrame()
        if sep is not None:
            return pd.read_csv(path, sep=sep, dtype=str, engine="python", header=None)
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
                return pd.read_csv(path, sep=s, dtype=str, engine="python", header=None)

        # fallback: leer línea a línea como una sola columna (evita sep="\n")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = [ln.rstrip("\n\r") for ln in fh if ln.strip() != ""]
        except Exception:
            lines = []
        return pd.DataFrame(lines, dtype=str)

    df_txt = _read_flexible(txt_path, txt_sep)
    df_tsv = _read_flexible(tsv_path, tsv_sep)

    if df_txt.empty:
        result = df_tsv
    elif df_tsv.empty:
        result = df_txt
    else:
        if axis == 0:
            result = pd.concat([df_txt, df_tsv], axis=0, ignore_index=True, sort=False)
        else:
            result = pd.concat([df_txt.reset_index(drop=True), df_tsv.reset_index(drop=True)], axis=1)

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
    print("Ejecutando algoritmo DIAMOND...")

    results = []
    seeds = [s for s in seed_genes if s in G]
    disease_module = set(seed_genes)
    if(not seeds):
        print("Ninguna semilla encontrada: convierte la red a ENTREZ/HUGO o usa semillas presentes en la red.")
    else:
        print(f"Semillas encontradas: {seeds}")
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



def map_series_to_entrez(series, batch_size=500): #Mapear los simbolos HUGO a ID Entrez
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

def prompt_yes_no(question, default='y'):
    """Pregunta sí/no por consola. Devuelve True para sí, False para no."""
    d = (default or 'y').lower()
    hint = "[Y/n]" if d == 'y' else "[y/N]"
    try:
        r = input(f"{question} {hint} ").strip().lower()
    except Exception:
        return d == 'y'
    if r == "":
        return d == 'y'
    return r in ('y', 'yes')

def _resolve_path(path, descr):
        if path and os.path.exists(path):
            return path
        print(f"Fichero esperado para {descr} no encontrado: {path}")
        if prompt_yes_no(f"¿Desea introducir otra ruta para {descr}?", default='y'):
            new = input(f"Introduce la ruta para {descr} (enter para omitir): ").strip()
            return new if new else None
        print(f"Omitiendo {descr}.")
        return None
############################################
#################### MAIN ##################
############################################


 

def main(argv=None):
    parser = argparse.ArgumentParser(description="Pipeline DIAMOnD: map, build network, run DIAMOnD")
    parser.add_argument("--txt", default="data/network_diamond.txt", help="ruta del .txt de entrada")
    parser.add_argument("--tsv", default="data/string_network_filtered_hugo-400.tsv", help="ruta del .tsv de entrada")
    # parser.add_argument("--merged", default="results/merged_output.tsv", help="ruta TSV combinado de entrada/salida")
    parser.add_argument("--out-dir", default="results", help="directorio de salida")
    parser.add_argument("--seeds", default="ENO1,PGK1,HK2", help="semillas HUGO separadas por comas")
    parser.add_argument("--n-genes", type=int, default=100, help="número de genes a obtener con DIAMOnD")
    args = parser.parse_args(argv)

    # Preguntar al usuario si quieres por defecto o no 
    if prompt_yes_no("Desea usar las rutas/valores por defecto indicados en los argumentos (--txt/--tsv/--merged/--out-dir)?", default='y'):
        txt_path = args.txt
        tsv_path = args.tsv
       # merged_path = args.merged
        out_dir = args.out_dir
        seed_arg = args.seeds
        n_genes = args.n_genes
    else:
        txt_path = input(f"Ruta .txt [{args.txt}]: ").strip() or args.txt
        tsv_path = input(f"Ruta .tsv [{args.tsv}]: ").strip() or args.tsv
       # merged_path = input(f"Ruta merged output [{args.merged}]: ").strip() or args.merged
        out_dir = input(f"Directorio de salida [{args.out_dir}]: ").strip() or args.out_dir
        seed_arg = input(f"Semillas HUGO separadas por comas [{args.seeds}]: ").strip() or args.seeds
        n_genes_str = input(f"Número de genes DIAMOnD [{args.n_genes}]: ").strip()
        try:
            n_genes = int(n_genes_str) if n_genes_str else args.n_genes
        except ValueError:
            print("Valor inválido para n-genes, usando valor por defecto.")
            n_genes = args.n_genes

    os.makedirs(out_dir, exist_ok=True)

    # Ejecutar pipeline con las rutas/valores seleccionados
    merge_txt_and_tsv(txt_path, tsv_path, "results/merged_output.tsv")

    df = pd.read_csv("results/merged_output.tsv", header=None, names=["Protein1", "Protein2"], sep="\t", dtype=str, engine="python")

    df["Protein1"] = map_series_to_entrez(df["Protein1"])
    df["Protein2"] = map_series_to_entrez(df["Protein2"])

    df = df[(df["Protein1"] != "") & (df["Protein2"] != "")].copy()
    df.to_csv(os.path.join(out_dir, "merged_output_entrez.tsv"), sep="\t", index=False)

    G = nx.from_pandas_edgelist(df, source="Protein1", target="Protein2")
    print(f"Red cargada con {G.number_of_nodes()} nodos y {G.number_of_edges()} interacciones.")

    seed_list = [s.strip() for s in seed_arg.split(",") if s.strip()]
    gene_ids = get_gene_ids_online(seed_list)
    print("Mapping:", gene_ids)

    nodes = set(G.nodes())
    seed_genes = []
    for sym, entrez in gene_ids.items():
        if entrez and str(entrez) in nodes:
            seed_genes.append(str(entrez))
        elif sym in nodes:
            seed_genes.append(sym)

    if not seed_genes:
        print("Ninguna semilla encontrada en la red.")
        return 1

    print(f"Semillas encontradas: {seed_genes}")
    module_df = algoritmo_diamond(G, seed_genes, n_genes=n_genes)
    out_module = os.path.join(out_dir, "diamond_module_results.csv")
    module_df.to_csv(out_module, index=False)
    print("Resultados guardados en", out_module)
    print(module_df.head())

if __name__ == "__main__":
    sys.exit(main())