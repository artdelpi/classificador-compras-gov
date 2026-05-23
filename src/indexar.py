import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "modelo_retrieval")
BASE_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 512


def main():
    tabela_path = os.path.join(DATASET_DIR, "tabela_ncm.csv.gz")
    for enc in ("utf-8", "latin-1"):
        try:
            tabela = pd.read_csv(tabela_path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue

    tabela["ncm_8dig"] = tabela["codigo"].apply(
        lambda x: str(int(float(x))).zfill(8) if pd.notna(x) else None
    )
    tabela = tabela.dropna(subset=["ncm_8dig"])
    tabela["ncm_2dig"] = tabela["ncm_8dig"].str[:2]
    tabela["ncm_4dig"] = tabela["ncm_8dig"].str[:4]

    # nível capítulo: um texto por capítulo distinto
    capitulos = (
        tabela[tabela["nivel"].str.strip() == "Capítulo"]
        [["ncm_2dig", "capitulo_descricao"]]
        .drop_duplicates("ncm_2dig")
        .reset_index(drop=True)
    )
    capitulos["texto"] = capitulos["capitulo_descricao"]

    # nível posição: um texto por posição (4 dígitos)
    posicoes = (
        tabela[tabela["nivel"].str.strip() == "Posição"]
        [["ncm_4dig", "ncm_2dig", "descricao"]]
        .drop_duplicates("ncm_4dig")
        .reset_index(drop=True)
    )
    posicoes["texto"] = posicoes["descricao"]

    # nível subitem: um texto por NCM folha (8 dígitos), usa caminho completo
    subitens = (
        tabela[tabela["nivel"].str.strip() == "Subitem"]
        [["ncm_8dig", "ncm_4dig", "ncm_2dig", "descricao",
          "capitulo_codigo", "capitulo_descricao", "caminho"]]
        .drop_duplicates("ncm_8dig")
        .reset_index(drop=True)
    )
    subitens["capitulo_codigo"] = subitens["capitulo_codigo"].apply(
        lambda x: str(int(float(x))).zfill(2) if pd.notna(x) else None
    )
    subitens["texto"] = subitens["caminho"].fillna(subitens["descricao"])

    model_path = MODEL_DIR if os.path.isdir(MODEL_DIR) else BASE_MODEL
    print(f"Carregando encoder de: {model_path}")
    model = SentenceTransformer(model_path)

    def embed(textos, label):
        print(f"Gerando embeddings - {label}: {len(textos):,} itens")
        return model.encode(
            textos,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

    emb_cap = embed(capitulos["texto"].tolist(), "capítulos")
    emb_pos = embed(posicoes["texto"].tolist(),  "posições")
    emb_sub = embed(subitens["texto"].tolist(),  "subitens")

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_DIR, "emb_capitulos.npy"), emb_cap)
    np.save(os.path.join(MODEL_DIR, "emb_posicoes.npy"),  emb_pos)
    np.save(os.path.join(MODEL_DIR, "emb_subitens.npy"),  emb_sub)

    capitulos.drop(columns="texto").to_parquet(os.path.join(MODEL_DIR, "meta_capitulos.parquet"), index=False)
    posicoes.drop(columns="texto").to_parquet(os.path.join(MODEL_DIR, "meta_posicoes.parquet"),  index=False)
    subitens.drop(columns="texto").to_parquet(os.path.join(MODEL_DIR, "meta_subitens.parquet"),  index=False)

    print(f"\nDimensão dos embeddings: {emb_sub.shape[1]}")
    print(f"Capítulos indexados: {len(capitulos)}")
    print(f"Posições indexadas:  {len(posicoes)}")
    print(f"Subitens indexados:  {len(subitens)}")


if __name__ == "__main__":
    main()
