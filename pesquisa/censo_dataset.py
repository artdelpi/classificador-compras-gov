"""Censo do dataset: quantifica a estrutura dos rótulos conflitantes.

Produz os números citados na issue #2 e no RELATORIO.md: descrições
multi-código, teto oracle, alinhamento por família NCM e segregação por fonte.
Roda em segundos, sem modelo. Uso: python pesquisa/censo_dataset.py
"""
import os

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    df = pd.read_csv(os.path.join(RAIZ, "dataset", "dataset.csv.gz"),
                     dtype=str).dropna(subset=["descricao_catmat", "ncm_8dig"])
    df["ncm_4dig"] = df["ncm_8dig"].str[:4]
    sub = pd.read_parquet(os.path.join(RAIZ, "modelo_retrieval", "meta_subitens.parquet"))
    familia = sub.groupby("ncm_4dig")["ncm_8dig"].nunique()

    g = df.groupby("descricao_catmat")["ncm_8dig"].agg(set)
    multi = g[g.apply(len) > 1]
    frac_linhas = df["descricao_catmat"].isin(multi.index).mean()
    print(f"Descrições com >1 código: {len(multi):,} de {len(g):,} únicas "
          f"({100*frac_linhas:.1f}% das {len(df):,} linhas)")
    n_cod = multi.apply(len)
    print(f"Códigos distintos por descrição multi-código: p50={n_cod.median():.0f} "
          f"média={n_cod.mean():.1f} máx={n_cod.max()}")

    def analisa(codigos):
        pos = pd.Series([c[:4] for c in codigos])
        cap = pd.Series([c[:2] for c in codigos])
        pos_dom = pos.value_counts().idxmax()
        fam = familia.get(pos_dom, 0)
        cobertura = sum(1 for c in codigos if c[:4] == pos_dom) / fam if fam else 0
        return (pos.value_counts(normalize=True).iloc[0],
                cap.value_counts(normalize=True).iloc[0], cobertura)

    r = multi.apply(lambda s: pd.Series(analisa(s), index=["share_pos", "share_cap", "cobertura"]))
    print(f"\nEstrutura de família nas descrições multi-código:")
    print(f"  100% dos códigos na mesma posição (4 díg.): {100*(r['share_pos']==1).mean():.1f}%")
    print(f"  100% dos códigos no mesmo capítulo (2 díg.): {100*(r['share_cap']==1).mean():.1f}%")
    print(f"  cobrem >=80% da família da posição dominante: {100*(r['cobertura']>=.8).mean():.1f}%")

    maj = df.groupby("descricao_catmat")["ncm_8dig"].agg(lambda s: s.value_counts().idxmax())
    acerto = (df["ncm_8dig"] == df["descricao_catmat"].map(maj)).mean()
    maj4 = df.groupby("descricao_catmat")["ncm_4dig"].agg(lambda s: s.value_counts().idxmax())
    acerto4 = (df["ncm_4dig"] == df["descricao_catmat"].map(maj4)).mean()
    print(f"\nTeto oracle (rótulo majoritário por descrição exata): "
          f"8 díg = {100*acerto:.1f}% | 4 díg = {100*acerto4:.1f}%")

    residual = df["ncm_8dig"].str[-2:].isin(["90", "99", "00"]).mean()
    print(f"Rótulos em código residual (90/99/00): {100*residual:.1f}%")

    cont = df["ncm_8dig"].value_counts()
    sem_exemplo = sub[~sub["ncm_8dig"].isin(cont.index)].shape[0]
    print(f"Cauda: p50={cont.median():.0f} exemplos/NCM | NCMs do índice sem exemplo: "
          f"{sem_exemplo:,} de {len(sub):,} | linhas em NCMs com >=5 exemplos: "
          f"{100*df['ncm_8dig'].map(cont).ge(5).mean():.1f}%")

    df["multi"] = df["descricao_catmat"].isin(multi.index)
    print("\nLinhas multi-código por fonte:")
    for fonte, gr in df.groupby("fonte", dropna=False):
        print(f"  {str(fonte):20s} {100*gr['multi'].mean():5.1f}%  (n={len(gr):,})")


if __name__ == "__main__":
    main()
