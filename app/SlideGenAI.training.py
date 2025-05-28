 # SlideGenAI Training Script with AMP Fixes, Padding Fix, and Scheduler Order

import os
import json
import torch
import random
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from rouge_score import rouge_scorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("🚨 GPU not detected. Please install PyTorch with CUDA support.")

print("✅ Using device:", torch.cuda.get_device_name(0))

MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

ARXIV_PATH = r"C:\Users\amira\Downloads\datasets SlidegenAI\arxiv-dataset\arxiv-dataset\train.txt"
PUBMED_PATH = r"C:\Users\amira\Downloads\datasets SlidegenAI\pubmed-dataset\pubmed-dataset\train.txt"

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def filter_sample(sample):
    return "article_text" in sample and len(sample["article_text"]) > 0 and len(" ".join(sample["article_text"])) > 200

def split_into_slides(text_list, n_slides=4):
    chunk_size = len(text_list) // n_slides + 1
    return [' '.join(text_list[i:i + chunk_size]).strip() for i in range(0, len(text_list), chunk_size)]

def preprocess(samples, name):
    result = []
    for s in samples:
        if filter_sample(s):
            slides = split_into_slides(s["article_text"])
            result.append({
                "abstract": " ".join(s["article_text"]),
                "slides": " ".join(slides)
            })
    print(f"{name} loaded: {len(result)} samples")
    return result

class SlideDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=512, max_output_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_enc = self.tokenizer(item["abstract"], truncation=True, padding="max_length", max_length=self.max_input_len, return_tensors="pt")
        target_enc = self.tokenizer(item["slides"], truncation=True, padding="max_length", max_length=self.max_output_len, return_tensors="pt")
        labels = target_enc["input_ids"].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100  # Mask padding
        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }

arxiv_data = preprocess(load_jsonl(ARXIV_PATH), "arXiv")
pubmed_data = preprocess(load_jsonl(PUBMED_PATH), "PubMed")
all_data = arxiv_data + pubmed_data

train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
train_data = train_data[:1024]  # Debug subset
val_data = val_data[:256]

train_dataset = SlideDataset(train_data, tokenizer)
val_dataset = SlideDataset(val_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * 3)
scaler = torch.amp.GradScaler(device_type="cuda")
writer = SummaryWriter("runs/SlideGenAI")

def train_model(epochs=3):
    model.train()
    loss_list = []

    for epoch in range(epochs):
        total_loss = 0
        print(f"\n🚀 Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(tqdm(train_loader)):
            start_time = time.time()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            global_step = epoch * len(train_loader) + step
            writer.add_scalar("Train/Loss", loss.item(), global_step)

            elapsed = time.time() - start_time
            if step % 50 == 0:
                print(f"Step {step} - Loss: {loss.item():.4f} - Time: {elapsed:.2f}s")

        avg_loss = total_loss / len(train_loader)
        loss_list.append(avg_loss)
        print(f"✅ Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "flan_t5_slidegen.pth")
    print("💾 Model saved to flan_t5_slidegen.pth")

    plt.figure()
    plt.plot(range(1, epochs + 1), loss_list, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss_plot.png")
    print("📊 Loss plot saved to training_loss_plot.png")

def evaluate_model(n_samples=20):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []

    with torch.no_grad():
        for i in range(min(n_samples, len(val_dataset))):
            item = val_dataset[i]
            input_text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
            target_text = tokenizer.decode(item["labels"], skip_special_tokens=True)

            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            rouge = scorer.score(target_text, prediction)
            scores.append(rouge)

    avg_r1 = np.mean([s['rouge1'].fmeasure for s in scores])
    avg_r2 = np.mean([s['rouge2'].fmeasure for s in scores])
    avg_rl = np.mean([s['rougeL'].fmeasure for s in scores])

    print(f"📈 ROUGE-1: {avg_r1:.4f}, ROUGE-2: {avg_r2:.4f}, ROUGE-L: {avg_rl:.4f}")

if __name__ == "__main__":
    train_model(epochs=3)
    evaluate_model(n_samples=20)
