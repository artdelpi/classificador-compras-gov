import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# carrega dados 
df = pd.read_csv("catmat_dataset.csv")

# remove descrições vazias
df = df.dropna(subset=["descricao", "grupo"])
df["descricao"] = df["descricao"].str.strip()

print(f"Total de itens: {len(df)}")
print(f"Grupos únicos: {df['grupo'].nunique()}")

# encode das labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["grupo"])
NUM_CLASSES = len(le.classes_)
print(f"Classes: {list(le.classes_)}")

df = df.groupby("grupo").filter(lambda x: len(x) >= 2)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print(f"\nTreino: {len(train_df)} | Teste: {len(test_df)}")

# tokenizador 
tokenizer = AutoTokenizer.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    do_lower_case=False
)

# dataset 
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

train_dataset = ComprasDataset(train_df["descricao"].tolist(), train_df["label"].tolist())
test_dataset  = ComprasDataset(test_df["descricao"].tolist(),  test_df["label"].tolist())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=8)

# modelo 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsando: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    num_labels=NUM_CLASSES
)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# treino 
EPOCHS = 3
print("\n--- INICIANDO TREINO ---")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(
            input_ids=      batch["input_ids"].to(device),
            attention_mask= batch["attention_mask"].to(device),
            labels=         batch["label"].to(device)
        )
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()

        if i % 100 == 0:
            print(f"  Época {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {outputs.loss.item():.4f}")

    print(f"Época {epoch+1} concluída | Loss médio: {total_loss/len(train_loader):.4f}")

# avaliação 
print("\n--- AVALIANDO ---")
model.eval()
preds, labels = [], []

with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            input_ids=      batch["input_ids"].to(device),
            attention_mask= batch["attention_mask"].to(device)
        )
        preds  += torch.argmax(outputs.logits, dim=1).cpu().tolist()
        labels += batch["label"].tolist()

print(classification_report(labels, preds, zero_division=0))

import os
os.makedirs("./modelo_catmat", exist_ok=True)
model.save_pretrained("./modelo_catmat")
tokenizer.save_pretrained("./modelo_catmat")

import pickle
with open("./modelo_catmat/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nModelo salvo em ./modelo_catmat")
