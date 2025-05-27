# SlideGenAI Testing Script for Final Project

import os
import csv
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/flan-t5-large"
MODEL_PATH = "flan_t5_slidegen.pth"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded and ready for testing.")

# ======= TEST SETUP =======
TEST_INPUTS = [
    "Deep learning is transforming medical image analysis, particularly in segmentation and disease classification.",
    "Climate change poses significant threats to biodiversity, food security, and global health.",
    "Quantum computing leverages principles of quantum mechanics to solve problems beyond classical limits.",
    "Recent advances in natural language processing have enabled chatbots to hold realistic conversations.",
    "Autonomous vehicles rely on sensor fusion, computer vision, and real-time decision-making algorithms.",
]

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Prepare output
results = []
for idx, abstract in enumerate(tqdm(TEST_INPUTS, desc="Generating Slides")):
    # Tokenize input
    inputs = tokenizer(abstract, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Generate slides
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    slides = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Dummy reference (replace with real reference if available)
    reference = abstract  # In real case: use expert-written slides

    # ROUGE evaluation
    score = scorer.score(reference, slides)
    r1 = score["rouge1"].fmeasure
    r2 = score["rouge2"].fmeasure
    rl = score["rougeL"].fmeasure

    print(f"\nInput {idx+1}: {abstract}")
    print(f"Generated Slides:\n{slides}")
    print(f"ROUGE-1: {r1:.4f}, ROUGE-2: {r2:.4f}, ROUGE-L: {rl:.4f}")

    results.append({
        "Input Abstract": abstract,
        "Generated Slides": slides,
        "ROUGE-1": f"{r1:.4f}",
        "ROUGE-2": f"{r2:.4f}",
        "ROUGE-L": f"{rl:.4f}"
    })

# Save results to CSV
output_csv = "slidegen_test_results.csv"
with open(output_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Test results saved to '{output_csv}'")
