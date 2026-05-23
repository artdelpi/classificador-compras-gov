# Classificador NCM para Compras Públicas

**DISCLAIMER**: esse é um projeto pessoal que não funciona perfeitamente e está em aprimoramento. Ir na seção **"Estado atual do modelo"** pra ver o desempenho atual.

A proposta desse repositório é oferecer um modelo de embeddings fine-tunado que gera vetores próximos pra descrições de CATMATS e NCMs correspondentes, sugerindo o código de NCM mais provável. Montei o dataset compilando dados do Ministério da Saúde e extração da API do comprasgov. 

Como a maioria dos mapeamentos era de itens relacionados à saúde, precisei gerar uma minoria de dados sintéticos com o llama:7b pra que CATMATS de outras categorias tivessem algum tipo de representatividade no treino!!

## Como funciona

1. Um modelo de embeddings (BGE-M3 fine-tunado) transforma descrições em vetores
2. A descrição de um CATMAT é vetorizada e comparada contra todos os subitens NCM (mais de 10k possibilidades...)
3. O resultado é uma lista rankeada de NCMs com score de similaridade

## Estado atual do modelo

| Métrica | Valor |
|---|---|
| Capítulo P@1 (2 dígitos)| 95.1% |
| Posição P@1 (4 dígitos) | 90.0% |
| Subitem P@1 (8 dígitos) | 32.8% |
| Subitem R@5 (8 dígitos) | 51.5% |

O modelo lista até 5 possibilidades mais prováveis de NCM para capítulo, posição e subitem, dada uma descrição de CATMAT. Ele, atualmente, é excelente em detectar o capítulo (95.1% de acerto considerando unicamente o capítulo mais provável) e posição(90%); o gargalo é obter o subitem, que tem mais de 10k possibilidades.

## Pesos do modelo

Os pesos não estão no repositório pq são muito pesados. Para usar o classificador você tem duas opções:

1. **Treinar do zero** rodando o fluxo de retreino abaixo (treinei usando uma RTX 5090 com 32GB de VRAM).
2. **Entrar em contato com o email artdelpi@gmail.com**, que eu disponibilizo os pesos num link do onedrive.

Os pesos devem ficar em `modelo_retrieval/`. Além do `model.safetensors` e configs do modelo, a pasta precisa dos arquivos `.npy` de índice gerados pelo `indexar.py`.

## Retreino (opcional)

```bash
python src/treinar.py # fine-tuna o modelo
python src/indexar.py # indexa os NCMs com o modelo novo
python src/avaliar.py # mede as métricas
```

## Como rodar

Dado que já tem os pesos do modelo treinado, faça:

Instala as dependências:

```bash
pip install -r requirements.txt
```

Classificador interativo:

```bash
python src/classificar.py
```

Avaliação do modelo no dataset de teste:

```bash
python src/avaliar.py
```

## Dataset

Os arquivos estão comprimidos em gzip:

- `dataset/dataset.csv.gz`: pares (descrição CATMAT, NCM) usados no treino
- `dataset/negativos.csv.gz`: triplas com hard negatives para o treino
- `dataset/tabela_ncm.csv.gz`: tabela NCM completa com hierarquia e caminhos
