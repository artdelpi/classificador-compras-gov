import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from classificar import ClassificadorNCM

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")

TOP_K = 5
TEST_SIZE = 0.2
BATCH_SIZE = 64


def recuperar_lote(clf, embs_tensor, top_k):
    resultados = []
    for i in tqdm(range(len(embs_tensor)), desc="Retrieval hierárquico"):
        resultados.append(clf.classificar_emb(embs_tensor[i:i+1], top_k))
    return resultados


def metricas(corretos_8, resultados, top_k):
    hits_1_8, hits_k_8 = [], []
    hits_1_4, hits_k_4 = [], []
    hits_1_2, hits_k_2 = [], []
    rr = []
    scores_acerto, scores_erro = [], []

    for ncm8, res in zip(corretos_8, resultados):
        ncm4 = ncm8[:4]
        ncm2 = ncm8[:2]

        rec_8 = [r["ncm_8dig"]      for r in res]
        rec_4 = [r["ncm_8dig"][:4]  for r in res]
        rec_2 = [r["ncm_8dig"][:2]  for r in res]

        hits_1_8.append(int(bool(rec_8) and rec_8[0] == ncm8))
        hits_k_8.append(int(ncm8 in rec_8))

        hits_1_4.append(int(bool(rec_4) and rec_4[0] == ncm4))
        hits_k_4.append(int(ncm4 in rec_4))

        hits_1_2.append(int(bool(rec_2) and rec_2[0] == ncm2))
        hits_k_2.append(int(ncm2 in rec_2))

        rank = next((r + 1 for r, n in enumerate(rec_8) if n == ncm8), None)
        rr.append(1 / rank if rank else 0)

        if res:
            score_top1 = res[0]["score"]
            if rec_8[0] == ncm8:
                scores_acerto.append(score_top1)
            else:
                scores_erro.append(score_top1)

    return {
        "P@1_8": np.mean(hits_1_8), "R@k_8": np.mean(hits_k_8),
        "P@1_4": np.mean(hits_1_4), "R@k_4": np.mean(hits_k_4),
        "P@1_2": np.mean(hits_1_2), "R@k_2": np.mean(hits_k_2),
        "MRR":         np.mean(rr),
        "hits_1_8":    hits_1_8,
        "hits_k_8":    hits_k_8,
        "scores_acerto": scores_acerto,
        "scores_erro":   scores_erro,
    }


def main():
    df = pd.read_csv(
        os.path.join(DATASET_DIR, "dataset.csv.gz"),
        dtype={"ncm_8dig": str, "ncm_2dig": str, "ncm_4dig": str, "ncm_6dig": str},
    ).dropna(subset=["descricao_catmat", "ncm_8dig", "ncm_2dig"])

    contagem = df["ncm_2dig"].value_counts()
    df = df[df["ncm_2dig"].isin(contagem[contagem >= 2].index)]
    _, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42, stratify=df["ncm_2dig"])
    print(f"Teste: {len(test_df):,} exemplos | {test_df['ncm_8dig'].nunique()} NCMs únicos")

    clf = ClassificadorNCM()
    print("\nEncoding test set...")
    embs = clf.model.encode(
        test_df["descricao_catmat"].tolist(),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embs_tensor = torch.from_numpy(embs.astype(np.float32)).to(clf.device)

    resultados_lote = recuperar_lote(clf, embs_tensor, TOP_K)
    res = metricas(test_df["ncm_8dig"].tolist(), resultados_lote, TOP_K)

    acertos   = sum(res["hits_1_8"])
    borderline = sum(1 for h1, hk in zip(res["hits_1_8"], res["hits_k_8"]) if not h1 and hk)
    falhas    = sum(1 for h1, hk in zip(res["hits_1_8"], res["hits_k_8"]) if not h1 and not hk)

    print(f"\n=== MÉTRICAS (top-1 / top-{TOP_K}) ===")
    print(f"  Capítulo  (2 dígitos):  P@1 = {res['P@1_2']*100:.1f}%   R@{TOP_K} = {res['R@k_2']*100:.1f}%")
    print(f"  Posição   (4 dígitos):  P@1 = {res['P@1_4']*100:.1f}%   R@{TOP_K} = {res['R@k_4']*100:.1f}%")
    print(f"  Subitem   (8 dígitos):  P@1 = {res['P@1_8']*100:.1f}%   R@{TOP_K} = {res['R@k_8']*100:.1f}%")
    print(f"  MRR (8 dígitos):        {res['MRR']:.4f}")
    print(f"\n  Acertos exatos top-1:   {acertos:,}")
    print(f"  Borderline (top-{TOP_K} ok):  {borderline:,}")
    print(f"  Falhas totais (top-{TOP_K}):  {falhas:,}")

    sa = np.array(res["scores_acerto"])
    se = np.array(res["scores_erro"])
    print("\n=== DISTRIBUIÇÃO DE SCORES (top-1) ===")
    print(f"  Acertos  — média: {sa.mean():.4f}  p25: {np.percentile(sa,25):.4f}  p50: {np.percentile(sa,50):.4f}  p75: {np.percentile(sa,75):.4f}")
    print(f"  Erros    — média: {se.mean():.4f}  p25: {np.percentile(se,25):.4f}  p50: {np.percentile(se,50):.4f}  p75: {np.percentile(se,75):.4f}")

    # thresholds candidatos e precisão em cada um
    print("\n=== THRESHOLD vs COBERTURA/PRECISÃO (8 dígitos) ===")
    print(f"  {'Threshold':>10}  {'Cobertura':>10}  {'Precisão':>10}")
    todos_scores = list(zip(res["scores_acerto"] + res["scores_erro"],
                            [1]*len(res["scores_acerto"]) + [0]*len(res["scores_erro"])))
    total = len(todos_scores)
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        acima = [(s, h) for s, h in todos_scores if s >= thr]
        if not acima:
            continue
        cobertura = len(acima) / total
        precisao  = sum(h for _, h in acima) / len(acima)
        print(f"  {thr:>10.2f}  {cobertura:>9.1%}  {precisao:>9.1%}")


if __name__ == "__main__":
    main()
