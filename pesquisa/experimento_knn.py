"""Experimento: retrieval por exemplos (kNN) vs retrieval por texto legal (baseline).

Compara o classificador atual com um kNN sobre as descrições CATMAT já
classificadas do conjunto de treino, usando o mesmo encoder fine-tunado.
Resultados e discussão: pesquisa/RELATORIO.md e issue #2.

Uso (da raiz do repositório, com os pesos em modelo_retrieval/):
    python pesquisa/experimento_knn.py

O encode do treino é salvo em chunks retomáveis em modelo_retrieval/knn_cache/
(~163 MB ao final; ~9,5h em CPU, minutos em GPU). Reexecuções aproveitam o cache.
"""
import os
import sys
import time
from collections import defaultdict

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from classificar import ClassificadorNCM, FLAT_N, BEAM_WIDTH

DATASET = os.path.join(RAIZ, "dataset", "dataset.csv.gz")
CACHE = os.path.join(RAIZ, "modelo_retrieval", "knn_cache")
TOP_K = 5
TEST_SIZE = 0.2
SEED = 42
CHUNK = 2000
K_VIZINHOS = 10
N_AMOSTRA = 2000


class ClassificadorEval(ClassificadorNCM):
    """Reintroduz o classificar_emb (retrieval de subitens a partir de um
    embedding pronto), ausente do classificar.py atual — mesma lógica do
    trecho de subitens do classificar()."""

    def classificar_emb(self, emb: torch.Tensor, top_k: int):
        flat_idxs, flat_scores = self._top_k(emb, self.emb_sub, min(FLAT_N, len(self.meta_sub)))
        candidatos = [
            {"ncm_8dig": self.meta_sub.iloc[i]["ncm_8dig"], "score": float(s)}
            for i, s in zip(flat_idxs, flat_scores)
        ]
        cap_candidatos = list(dict.fromkeys(
            self.meta_sub.iloc[i]["ncm_2dig"] for i in flat_idxs
        ))[:BEAM_WIDTH]
        for cap_code in cap_candidatos:
            mask_pos = self.meta_pos["ncm_2dig"].values == cap_code
            pos_embs = self.emb_pos[mask_pos]
            pos_meta = self.meta_pos[mask_pos].reset_index(drop=True)
            if len(pos_meta) == 0:
                continue
            for pos_idx, _ in zip(*self._top_k(emb, pos_embs, BEAM_WIDTH)):
                pos_code = pos_meta.iloc[pos_idx]["ncm_4dig"]
                mask_sub = self.meta_sub["ncm_4dig"].values == pos_code
                sub_embs = self.emb_sub[mask_sub]
                sub_meta = self.meta_sub[mask_sub].reset_index(drop=True)
                if len(sub_meta) == 0:
                    continue
                for sub_idx, score in zip(*self._top_k(emb, sub_embs, top_k)):
                    candidatos.append({
                        "ncm_8dig": sub_meta.iloc[sub_idx]["ncm_8dig"],
                        "score": float(score),
                    })
        vistos, subitens = set(), []
        for c in sorted(candidatos, key=lambda x: x["score"], reverse=True):
            if c["ncm_8dig"] not in vistos:
                vistos.add(c["ncm_8dig"])
                subitens.append(c)
            if len(subitens) >= top_k:
                break
        return subitens


def metricas(y_true, preds_topk):
    out = {}
    for nd in (2, 4, 8):
        out[f"P@1_{nd}"] = np.mean([bool(p) and p[0][:nd] == t[:nd] for t, p in zip(y_true, preds_topk)])
        out[f"R@5_{nd}"] = np.mean([t[:nd] in [x[:nd] for x in p] for t, p in zip(y_true, preds_topk)])
    rr = []
    for t, p in zip(y_true, preds_topk):
        rank = next((i + 1 for i, x in enumerate(p) if x == t), None)
        rr.append(1 / rank if rank else 0)
    out["MRR"] = np.mean(rr)
    return out


def imprime(nome, y_true, preds, mask_visto):
    print(f"\n=== {nome} ===")
    for rotulo, mask in [("GERAL", np.ones(len(y_true), bool)),
                         ("VISTO no treino", mask_visto),
                         ("NÃO visto", ~mask_visto)]:
        yt = [y for y, m in zip(y_true, mask) if m]
        pr = [p for p, m in zip(preds, mask) if m]
        m = metricas(yt, pr)
        print(f"  [{rotulo}] n={len(yt):,}  "
              f"cap P@1={m['P@1_2']*100:.1f}%  pos P@1={m['P@1_4']*100:.1f}%  "
              f"sub P@1={m['P@1_8']*100:.1f}%  sub R@5={m['R@5_8']*100:.1f}%  MRR={m['MRR']:.3f}")


def main():
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    df = pd.read_csv(
        DATASET, dtype={"ncm_8dig": str, "ncm_2dig": str, "ncm_4dig": str, "ncm_6dig": str},
    ).dropna(subset=["descricao_catmat", "ncm_8dig", "ncm_2dig"])
    contagem = df["ncm_2dig"].value_counts()
    df = df[df["ncm_2dig"].isin(contagem[contagem >= 2].index)]
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["ncm_2dig"])
    amostra = test_df.sample(n=N_AMOSTRA, random_state=SEED)

    # distribuição de rótulos por descrição única do treino
    grupos = train_df.groupby("descricao_catmat")["ncm_8dig"].value_counts()
    dist = defaultdict(list)
    for (desc, ncm), c in grupos.items():
        dist[desc].append((ncm, c))
    for desc in dist:
        total = sum(c for _, c in dist[desc])
        dist[desc] = [(n, c / total) for n, c in dist[desc]]
    descs_treino = sorted(dist.keys())
    print(f"Treino: {len(train_df):,} linhas | {len(descs_treino):,} descrições únicas", flush=True)

    visto = amostra["descricao_catmat"].isin(dist).values
    y_true = amostra["ncm_8dig"].tolist()

    clf = ClassificadorEval()

    teste_npy = os.path.join(CACHE, "teste_emb.npy")
    if os.path.exists(teste_npy):
        emb_teste = np.load(teste_npy)
    else:
        print("Encoding amostra de teste...", flush=True)
        emb_teste = clf.model.encode(amostra["descricao_catmat"].tolist(), batch_size=32,
                                     normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=True).astype(np.float32)
        np.save(teste_npy, emb_teste)

    base_parquet = os.path.join(CACHE, "baseline_preds.parquet")
    if os.path.exists(base_parquet):
        preds_base = pd.read_parquet(base_parquet)["preds"].apply(list).tolist()
    else:
        emb_t = torch.from_numpy(emb_teste).to(clf.device)
        preds_base = []
        for i in tqdm(range(len(emb_t)), desc="Baseline (texto legal)"):
            res = clf.classificar_emb(emb_t[i:i+1], TOP_K)
            preds_base.append([r["ncm_8dig"] for r in res])
        pd.DataFrame({"preds": preds_base}).to_parquet(base_parquet)

    n_chunks = (len(descs_treino) + CHUNK - 1) // CHUNK
    partes = []
    for ci in range(n_chunks):
        arq = os.path.join(CACHE, f"treino_emb_{ci:03d}.npy")
        if not os.path.exists(arq):
            lote = descs_treino[ci * CHUNK:(ci + 1) * CHUNK]
            print(f"Encoding chunk {ci+1}/{n_chunks} ({len(lote)} descrições) "
                  f"[{(time.time()-t0)/60:.0f} min decorridos]", flush=True)
            e = clf.model.encode(lote, batch_size=32, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)
            np.save(arq, e)
        partes.append(np.load(arq, mmap_mode="r"))
    emb_treino = np.vstack(partes)
    print(f"Treino encodado: {emb_treino.shape} [{(time.time()-t0)/60:.0f} min]", flush=True)

    preds_knn = []
    emb_treino_t = torch.from_numpy(np.ascontiguousarray(emb_treino))
    emb_teste_t = torch.from_numpy(emb_teste)
    for i in tqdm(range(len(emb_teste_t)), desc="kNN"):
        sims = emb_treino_t @ emb_teste_t[i]
        top = sims.topk(K_VIZINHOS)
        placar = defaultdict(float)
        for idx, s in zip(top.indices.tolist(), top.values.tolist()):
            for ncm, share in dist[descs_treino[idx]]:
                placar[ncm] += s * share
        preds_knn.append(sorted(placar, key=placar.get, reverse=True)[:TOP_K])

    pd.DataFrame({"desc": amostra["descricao_catmat"].tolist(), "ncm_true": y_true,
                  "visto": visto, "preds_knn": preds_knn,
                  "preds_base": preds_base}).to_parquet(os.path.join(CACHE, "resultados.parquet"))

    imprime("BASELINE — retrieval por texto legal (modelo atual)", y_true, preds_base, visto)
    imprime(f"kNN k={K_VIZINHOS} — retrieval por exemplos do treino", y_true, preds_knn, visto)
    print(f"\nTempo total: {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
