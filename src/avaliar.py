import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse

# args
parser = argparse.ArgumentParser(description="Avalia o modelo de classificação de compras")
parser.add_argument("--modelo",  default="./modelo_catmat",   help="Caminho do modelo treinado")
parser.add_argument("--dados",   default="./dataset/catmat_dataset.csv", help="Caminho do dataset")
parser.add_argument("--batch",   default=8, type=int,         help="Batch size")
args = parser.parse_args()

# carrega dados
df = pd.read_csv(args.dados).dropna(subset=["descricao", "grupo"])
df = df.groupby("grupo").filter(lambda x: len(x) >= 2)

with open(f"{args.modelo}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

df["label"] = le.transform(df["grupo"])
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

print(f"Itens de teste: {len(test_df)}")
print(f"Grupos: {df['grupo'].nunique()}")

# dataset
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.modelo)
model     = AutoModelForSequenceClassification.from_pretrained(args.modelo).to(device)
model.eval()

class ComprasDataset(Dataset):
    def __init__(self, textos, labels):
        self.textos = textos
        self.labels = labels
    def __len__(self):
        return len(self.textos)
    def __getitem__(self, idx):
        enc = tokenizer(
            self.textos[idx],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

test_loader = DataLoader(
    ComprasDataset(test_df["descricao"].tolist(), test_df["label"].tolist()),
    batch_size=args.batch
)

# avaliação
print(f"\nUsando: {device}")
print("Avaliando...\n")

preds, labels = [], []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            input_ids=      batch["input_ids"].to(device),
            attention_mask= batch["attention_mask"].to(device)
        )
        preds  += torch.argmax(outputs.logits, dim=1).cpu().tolist()
        labels += batch["label"].tolist()

# report
labels_presentes = sorted(set(labels + preds))
nomes_presentes  = le.inverse_transform(labels_presentes)

print(classification_report(
    labels, preds,
    labels=labels_presentes,
    target_names=nomes_presentes,
    zero_division=0
))

erros = test_df.copy()
erros["pred"] = le.inverse_transform(preds)
erros = erros[erros["grupo"] != erros["pred"]]

print(f"Total de erros: {len(erros)} / {len(test_df)}")
print("\nErros mais frequentes:")
print(
    erros.groupby(["grupo", "pred"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(10)
    .to_string(index=False)
)