import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "modelo_retrieval")
BASE_MODEL = "BAAI/bge-m3"
FLAT_N = 300
BEAM_WIDTH = 10


class ClassificadorNCM:
    def __init__(self, model_dir: str = MODEL_DIR):
        model_path = model_dir if os.path.isdir(model_dir) else BASE_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = SentenceTransformer(model_path, device=self.device)

        self.emb_cap = torch.from_numpy(np.load(os.path.join(model_dir, "emb_capitulos.npy"))).to(self.device)
        self.emb_pos = torch.from_numpy(np.load(os.path.join(model_dir, "emb_posicoes.npy"))).to(self.device)
        self.emb_sub = torch.from_numpy(np.load(os.path.join(model_dir, "emb_subitens.npy"))).to(self.device)

        self.meta_cap = pd.read_parquet(os.path.join(model_dir, "meta_capitulos.parquet"))
        self.meta_pos = pd.read_parquet(os.path.join(model_dir, "meta_posicoes.parquet"))
        self.meta_sub = pd.read_parquet(os.path.join(model_dir, "meta_subitens.parquet"))

    def _top_k(self, query: torch.Tensor, embs: torch.Tensor, k: int):
        scores = (embs @ query.T).squeeze(1)
        k = min(k, len(scores))
        top = scores.topk(k)
        return top.indices.cpu().tolist(), top.values.cpu().tolist()

    def classificar(self, descricao: str, top_k: int = 5) -> dict:
        emb = torch.from_numpy(
            self.model.encode([descricao], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        ).to(self.device)

        # capítulos
        cap_idxs, cap_scores = self._top_k(emb, self.emb_cap, top_k)
        capitulos = [
            {
                "ncm_2dig":  self.meta_cap.iloc[i]["ncm_2dig"],
                "descricao": self.meta_cap.iloc[i]["capitulo_descricao"],
                "score":     round(float(s), 4),
            }
            for i, s in zip(cap_idxs, cap_scores)
        ]

        # posições: busca flat em todas
        pos_idxs, pos_scores = self._top_k(emb, self.emb_pos, top_k)
        posicoes = [
            {
                "ncm_4dig":  self.meta_pos.iloc[i]["ncm_4dig"],
                "descricao": self.meta_pos.iloc[i]["descricao"],
                "score":     round(float(s), 4),
            }
            for i, s in zip(pos_idxs, pos_scores)
        ]

        # subitens: flat search top-300, depois refinamento hierárquico
        flat_idxs, flat_scores = self._top_k(emb, self.emb_sub, min(FLAT_N, len(self.meta_sub)))
        candidatos = [
            {"ncm_8dig": self.meta_sub.iloc[i]["ncm_8dig"],
             "descricao": self.meta_sub.iloc[i]["caminho"],
             "score": float(s)}
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
                    row = sub_meta.iloc[sub_idx]
                    candidatos.append({"ncm_8dig": row["ncm_8dig"],
                                       "descricao": row["caminho"],
                                       "score": float(score)})

        vistos, subitens = set(), []
        for c in sorted(candidatos, key=lambda x: x["score"], reverse=True):
            if c["ncm_8dig"] not in vistos:
                vistos.add(c["ncm_8dig"])
                subitens.append({**c, "score": round(c["score"], 4)})
            if len(subitens) >= top_k:
                break

        return {"capitulos": capitulos, "posicoes": posicoes, "subitens": subitens}


def main():
    clf = ClassificadorNCM()
    print("\nClassificador NCM pronto. Digite uma descrição CATMAT (ou 'sair').\n")

    while True:
        descricao = input("Descrição: ").strip()
        if descricao.lower() in ("sair", "exit", "q"):
            break
        if not descricao:
            continue

        res = clf.classificar(descricao)

        print("\n── Capítulos ──")
        for r in res["capitulos"]:
            print(f"  [{r['ncm_2dig']}] score={r['score']}  {r['descricao']}")

        print("\n── Posições ──")
        for r in res["posicoes"]:
            print(f"  [{r['ncm_4dig']}] score={r['score']}  {r['descricao']}")

        print("\n── Subitens ──")
        for r in res["subitens"]:
            print(f"  [{r['ncm_8dig']}] score={r['score']}  {r['descricao']}")
        print()


if __name__ == "__main__":
    main()
