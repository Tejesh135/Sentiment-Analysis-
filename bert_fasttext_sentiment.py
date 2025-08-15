import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import logging
import os
from tqdm import tqdm
from datetime import datetime

# ---- Logger Setup ----
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# ---- Label Maps ----
LABEL_MAP = {1: 0, 2: 1}  # Map fastText labels to 0=negative, 1=positive
REV_LABEL_MAP = {0: "negative", 1: "positive"}

# ---- Dataset Class ----
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, clean_text=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.clean_text = clean_text

    def __len__(self):
        return len(self.texts)

    def clean(self, text):
        text = str(text).strip()
        if self.clean_text:
            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            text = ' '.join([w.strip() for w in text.split() if w.strip()])
        return text

    def __getitem__(self, idx):
        text = self.clean(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ---- Load fastText-format dataset files ----
def parse_ft_file(filename):
    texts = []
    labels = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("__label__"):
                first_space = line.find(" ")
                label_part = line[:first_space]
                text_part = line[first_space + 1 :]
                label = int(label_part.replace("__label__", ""))
                texts.append(text_part)
                labels.append(label)
    return texts, labels

# ---- Training Epoch ----
def train_epoch(model, device, data_loader, optimizer, scheduler, epoch, max_epochs):
    model.train()
    losses = []
    progress = tqdm(data_loader, desc=f"Train Epoch {epoch + 1}/{max_epochs}", unit="batch")
    for batch in progress:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        progress.set_postfix(loss=loss.item(), avg_loss=np.mean(losses))
    avg_loss = np.mean(losses)
    logger.info(f"Epoch {epoch + 1} training loss: {avg_loss:.4f}")
    return avg_loss

# ---- Evaluation ----
def evaluate(model, device, data_loader, epoch=None, mode="Validation"):
    model.eval()
    predictions = []
    true_labels = []
    progress = tqdm(data_loader, desc=f"{mode} Epoch {epoch + 1 if epoch is not None else ''}", unit="batch")
    with torch.no_grad():
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")
    cm = confusion_matrix(true_labels, predictions)
    cr = classification_report(true_labels, predictions, target_names=[REV_LABEL_MAP[i] for i in sorted(REV_LABEL_MAP)])
    logger.info(f"{mode} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{cr}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm, "classification_report": cr}

# ---- Save Model ----
def save_model(model, filepath, optimizer=None, scheduler=None, epoch=None, run_id=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "run_id": run_id,
    }
    if optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, filepath)
    logger.info(f"Model saved to {filepath}")

# ---- Load Model ----
def load_model(filepath, model, optimizer=None, scheduler=None):
    state = torch.load(filepath)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    epoch = state.get("epoch", None)
    run_id = state.get("run_id", None)
    logger.info(f"Loaded model from {filepath} (epoch: {epoch})")
    return model, optimizer, scheduler, epoch, run_id

# ---- Predict Function for Interactive Testing ----
def predict_sentiment(text, model, tokenizer, device, max_len=128):
    model.eval()
    encoding = tokenizer.encode_plus(
        str(text),
        max_length=max_len,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred].item()
        return REV_LABEL_MAP[pred], confidence

# ---- Main Function ----
def main():
    # Hardcoded parameters - update paths for your system
    train_file = r"C:\Users\poola\Downloads\Sentiment analysis\train_small.ft.txt"
    val_file = r"C:\Users\poola\Downloads\Sentiment analysis\test_small.ft.txt"
    batch_size = 16
    max_len = 128
    epochs = 3
    learning_rate = 2e-5
    save_dir = "./saved_models"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"bert_sentiment_{run_id}.pt")
    os.makedirs(save_dir, exist_ok=True)

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # Load tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_texts, train_labels_raw = parse_ft_file(train_file)
    val_texts, val_labels_raw = parse_ft_file(val_file)
    train_labels = [LABEL_MAP[x] for x in train_labels_raw]
    val_labels = [LABEL_MAP[x] for x in val_labels_raw]

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len, clean_text=True)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_len, clean_text=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model and optimizer
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABEL_MAP))
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs * len(train_loader))

    # Training loop
    best_f1 = 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, scheduler, epoch, epochs)
        logger.info(f"Epoch {epoch + 1} completed - Train Loss: {train_loss:.4f}")
        eval_metrics = evaluate(model, device, val_loader, epoch, mode="Validation")
        if eval_metrics["f1"] > best_f1:
            best_f1 = eval_metrics["f1"]
            save_model(model, save_path, optimizer, scheduler, epoch + 1, run_id)
            logger.info(f"New best F1 score: {best_f1:.4f}. Model saved.")

    # Final evaluation
    logger.info("Final Validation Evaluation:")
    evaluate(model, device, val_loader, mode="Validation")

    # Interactive mode for prediction
    logger.info("\nType a sentence to predict sentiment or 'quit' to exit.")
    while True:
        inp = input("Input: ").strip()
        if inp.lower() == "quit":
            break
        pred, conf = predict_sentiment(inp, model, tokenizer, device, max_len)
        print(f"Predicted: {pred} (Confidence: {conf:.4f})")

# Required function parse_ft_file used inside main()
def parse_ft_file(filename):
    texts = []
    labels = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("__label__"):
                first_space = line.find(" ")
                label_part = line[:first_space]
                text_part = line[first_space + 1 :]
                label = int(label_part.replace("__label__", ""))
                texts.append(text_part)
                labels.append(label)
    return texts, labels

if __name__ == "__main__":
    main()