import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# carrega modelo 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("./modelo_catmat")
model = AutoModelForSequenceClassification.from_pretrained("./modelo_catmat").to(device)
model.eval()

with open("./modelo_catmat/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def classificar(descricao: str) -> dict:
    enc = tokenizer(
        descricao,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=1).cpu()[0]
        idx    = probs.argmax().item()

    return {"categoria": le.inverse_transform([idx])[0], "confianca": float(probs[idx])}

if __name__ == "__main__":
    print("Modelo carregado. Digite descrições para classificar.")
    print("(digite 'sair' para encerrar)")
    while True:
        descricao = input("Descrição: ").strip()
        if descricao.lower() == "sair":
            break
        if descricao:
            resultado = classificar(descricao)
            print(f"Categoria: {resultado['categoria']}")
            print(f"Confiança: {resultado['confianca']*100:.1f}%")
