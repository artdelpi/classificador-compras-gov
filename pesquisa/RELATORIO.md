# Retrieval por exemplos (kNN) para o subitem NCM: experimento e diagnóstico

Relatório do experimento discutido na [issue #2](https://github.com/artdelpi/classificador-compras-gov/issues/2). Compara o retrieval atual (por texto legal do NCM) com retrieval por exemplos (kNN sobre descrições CATMAT já classificadas), usando o mesmo encoder fine-tunado do repositório, e documenta a causa raiz do teto de acurácia no subitem de 8 dígitos.

## 1. Metodologia

**Sistemas comparados** (ambos usam o mesmo encoder; a única diferença é o alvo da comparação):

- **Baseline (atual):** descrição comparada por similaridade de cosseno com o *texto legal* de cada código, com o refinamento hierárquico do `classificar.py` (flat top-300 + beam por capítulo/posição).
- **kNN (proposto):** descrição comparada com as **39.535 descrições CATMAT únicas do conjunto de treino**. Os k=10 vizinhos mais próximos votam: cada vizinho contribui, para cada código que carrega no histórico, com peso = similaridade × fração daquele código nos rótulos da descrição vizinha.

**Protocolo:** split 80/20 estratificado por capítulo (`random_state=42`, mesmo protocolo do `avaliar.py`); avaliação em amostra de 2.000 exemplos do teste (`.sample(2000, random_state=42)`); top-5; P@1, R@5 e MRR nos três níveis. A amostra é particionada em **descrições repetidas** (texto idêntico presente no treino, n=1.026) e **inéditas** (n=974).

## 2. Resultados

Subitem 8 dígitos, geral (n=2.000):

| Métrica | Baseline (texto legal) | kNN (exemplos) |
|---|---|---|
| P@1 | 31,4% | **42,4%** (+11,0 pp) |
| R@5 | 49,2% | 51,0% |
| MRR | 0,380 | **0,457** |

O kNN também melhora capítulo (P@1 93,3% → 97,3%) e posição (87,7% → 93,0%).

Por partição:

| Partição | Baseline | kNN | Teto oracle* |
|---|---|---|---|
| Repetidas (n=1.026) | P@1 4,4% · R@5 21,0% | P@1 1,9% · R@5 12,3% | **6,5%** |
| Inéditas (n=974) | P@1 59,8% · R@5 79,1% | **P@1 85,0% · R@5 91,9%** | ~100% |

\* Teto oracle: acurácia máxima de um classificador hipotético que respondesse o rótulo majoritário de cada descrição — limite imposto pelos *dados* (rótulos conflitantes para textos idênticos), não pelos modelos.

Na partição "repetidas" ambos os sistemas operam colados no teto de 6,5% — a diferença ali não é informativa. Na partição "inéditas" o kNN reduz o erro top-1 de 40,2% para 15,0%.

## 3. Causa raiz do teto: artefato de construção do dataset

Censo completo das 93.184 linhas (`censo_dataset.py`):

1. 3.505 descrições únicas (51,5% das linhas) associadas a mais de um NCM — mediana de 7, média de 13,7 e máximo de 71 códigos distintos por descrição (ponderando pela frequência das linhas, a descrição típica de uma linha conflitante carrega ~22 códigos).
2. O conflito tem estrutura de família: em **96,3%** dessas descrições todos os códigos pertencem à mesma posição de 4 dígitos (100% ao mesmo capítulo); em 50% cobrem ≥80% da família de subitens da posição. Ex.: uma mesma copiadora laser rotulada com todos os 71 subitens de 8443.xx.
3. A coluna `fonte` fecha o caso: **100,0% das 46.827 linhas de `fonte="cruzamento"` são multi-código**, contra 2,3% em `mapeamento_manual` e 3,5% em `sintetico` — o join CATMAT×NCM dessa fonte vinculou cada item à família inteira.

Consequências: as métricas agregadas de 8 dígitos subestimam qualquer modelo (metade da avaliação é impossível por construção), e o fine-tuning também treinou sobre essas linhas.

## 4. Limitações

- **"Inédita" = texto exatamente distinto.** Pode haver quase-duplicatas no treino, das quais o kNN se beneficia; reflete o uso real, mas um recorte por limiar de similaridade daria medida mais conservadora de generalização.
- **Amostra de 2.000** → margem de ~±2–3 pp; a diferença de +25,2 pp nas inéditas é muito superior, mas diferenças pequenas (ex.: R@5 geral) não são conclusivas.
- **k=10 e o esquema de voto não foram otimizados** — números do kNN são piso, não teto.
- **O kNN não cobre a cauda:** 7.113 dos 10.515 subitens do índice não têm exemplo no dataset; para eles o retrieval legal continua necessário (daí a proposta híbrida).

## 5. Como reproduzir

Requisitos: pesos fine-tunados em `modelo_retrieval/` (ver seção "Pesos do modelo" do README) e dependências do `requirements.txt` (em máquina sem GPU NVIDIA, instalar `torch` da variante CPU: `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu`).

```bash
python pesquisa/censo_dataset.py    # censo dos rótulos: segundos, sem modelo
python pesquisa/experimento_knn.py  # experimento completo
```

Todos os sorteios têm semente fixa (42). O encode do treino (etapa cara: ~9,5h em CPU, minutos em GPU) é salvo em chunks retomáveis em `modelo_retrieval/knn_cache/` (~163 MB ao final); reexecuções e interrupções aproveitam o cache. Ambiente usado: Python 3.13, `torch 2.7.1` (CPU), `sentence-transformers 5.5.1`.

## 6. Propostas

1. **Modo híbrido no `classificar.py`:** kNN quando há vizinhos no histórico, fallback para o índice legal na cauda sem exemplos, sem quebrar a API atual.
2. **Limpeza do dataset:** tratar as linhas `fonte="cruzamento"` (descartar, adjudicar refazendo o join na granularidade certa, ou rebaixar a supervisão fraca de nível posição) — melhora treino, avaliação e qualquer modelo futuro.
3. **Devolver distribuição histórica** (códigos e frequências) para descrições com vínculo legítimo a vários códigos, em vez de resposta única; a distribuição de votos dos vizinhos também é um substituto natural para o score como medida de confiança.
