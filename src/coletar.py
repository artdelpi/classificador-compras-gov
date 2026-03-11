import requests
import pandas as pd
import time

BASE = "https://dadosabertos.compras.gov.br"

def coletar_itens(max_paginas=500):
    todos_itens = []

    resp = requests.get(
        f"{BASE}/modulo-material/4_consultarItemMaterial",
        params={"pagina": 1, "tamanhoPagina": 100, "bps": False},
        headers={"accept": "*/*"}
    )

    if resp.status_code != 200:
        print(f"Erro na API: {resp.status_code}")
        return None

    dados = resp.json()
    total_paginas = min(dados["totalPaginas"], max_paginas)
    print(f"Total de itens na API: {dados['totalRegistros']}")
    print(f"Coletando {max_paginas} páginas de {dados['totalPaginas']} disponíveis\n")

    for pagina in range(1, total_paginas + 1):
        try:
            resp = requests.get(
                f"{BASE}/modulo-material/4_consultarItemMaterial",
                params={"pagina": pagina, "tamanhoPagina": 100, "bps": False},
                headers={"accept": "*/*"}
            )
            dados = resp.json()

            for item in dados.get("resultado", []):
                todos_itens.append({
                    "codigoItem":   item["codigoItem"],
                    "descricao":    item["descricaoItem"],
                    "codigoGrupo":  item["codigoGrupo"],
                    "grupo":        item["nomeGrupo"],
                    "codigoClasse": item["codigoClasse"],
                    "classe":       item["nomeClasse"],
                })

            if pagina % 10 == 0:
                print(f"Página {pagina}/{total_paginas} — {len(todos_itens)} itens")

            time.sleep(0.3)

        except Exception as e:
            print(f"Erro na página {pagina}: {e}")
            time.sleep(2)
            continue

    df = pd.DataFrame(todos_itens)
    df.to_csv("catmat_dataset.csv", index=False, encoding="utf-8")

    print(f"\n--- CONCLUÍDO ---")
    print(f"Total: {len(df)} itens")
    print(f"Grupos únicos: {df['grupo'].nunique()}")
    print(f"\nDistribuição por grupo:")
    print(df['grupo'].value_counts())

    return df

df = coletar_itens(max_paginas=500)