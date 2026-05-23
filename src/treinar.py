import os
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_IN = os.path.join(os.path.dirname(__file__), "..", "modelo_retrieval")   
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "modelo_retrieval")
BASE_MODEL = "BAAI/bge-m3"

BATCH_SIZE = 256        # batch efetivo 
MINI_BATCH = 32         # tamanho real do forward pass
EPOCHS = 5              # mais épocas
WARMUP_STEPS = 0.06
LR = 5e-6               # lr menor: fine-tuning a partir de modelo já treinado
MAX_SEQ_LEN = 256


def main():
    # carrega dataset: prefere triplas com hard negatives se disponível
    neg_path = os.path.join(DATASET_DIR, "negativos.csv.gz")
    if os.path.exists(neg_path):
        df = pd.read_csv(neg_path)
        print(f"Usando hard negatives: {len(df):,} triplas")
        train_dataset = Dataset.from_dict({
            "anchor":   df["anchor"].tolist(),
            "positive": df["positive"].tolist(),
            "negative": df["negative"].tolist(),
        })
    else:
        df = pd.read_csv(
            os.path.join(DATASET_DIR, "dataset.csv.gz"),
            dtype={"ncm_8dig": str, "ncm_2dig": str, "ncm_4dig": str, "ncm_6dig": str},
        ).dropna(subset=["descricao_catmat", "ncm_descricao"])
        df["texto_ncm"] = df["caminho"].fillna(df["ncm_descricao"])
        print(f"Usando pares simples: {len(df):,}")
        train_dataset = Dataset.from_dict({
            "anchor":   df["descricao_catmat"].tolist(),
            "positive": df["texto_ncm"].tolist(),
        })

    # inicia do modelo atual (já fine-tunado) 
    model_path = MODEL_IN if os.path.isdir(MODEL_IN) else BASE_MODEL
    print(f"Carregando modelo base de: {model_path}")
    model = SentenceTransformer(model_path, model_kwargs={"torch_dtype": "bfloat16"})
    model.max_seq_length = MAX_SEQ_LEN

    # CachedMultipleNegativesRankingLoss = GradCache
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH)

    total_steps = (len(train_dataset) // BATCH_SIZE) * EPOCHS
    print(f"\nSteps estimados: {total_steps:,} | Negativos por step: {BATCH_SIZE - 1}")
    print(f"Modelo de entrada: {model_path}")

    args = SentenceTransformerTrainingArguments(
        output_dir=MODEL_OUT,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        save_strategy="epoch",
        logging_steps=50,
        dataloader_drop_last=True, # evita batch menor que MINI_BATCH no fim da época
    )

    print("\n--- INICIANDO TREINO ---")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train()
    model.save_pretrained(MODEL_OUT)
    print(f"\nModelo salvo em: {MODEL_OUT}")


if __name__ == "__main__":
    main()
