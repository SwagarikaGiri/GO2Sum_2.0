import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from dataset import GODocDataset
from torch.utils.data import Dataset
from datasets import load_metric
import numpy as np

# === Configuration ===
print("Configuration setup...")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAIN_TSV = "data/train_dataset_cleaned.tsv"
VAL_TSV = "data/val_dataset_cleaned.tsv"
OBO_PATH = "data/gene_ontology.obo"
MAX_LENGTH = 512

# === Load tokenizer and model ===
print("Loading tokenizer and model from Hugging Face...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float32,  # changed from float16 to float32 for stability
    trust_remote_code=True
)
print("Model and tokenizer loaded successfully.")

# === Prepare datasets ===
print("Loading datasets...")
train_dataset = GODocDataset(TRAIN_TSV, OBO_PATH)
val_dataset = GODocDataset(VAL_TSV, OBO_PATH)
print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

# === Causal LM Tokenization with Target Truncation ===
def tokenize_example(example):
    prompt = "Summarize GO doc: " + str(example.get("input", "")).strip()
    target = str(example.get("target", "")).strip()

    if not prompt or not target or target.lower() == "nan":
        return None

    prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH // 2)["input_ids"]
    target_ids = tokenizer(target, truncation=True, max_length=MAX_LENGTH - len(prompt_ids))["input_ids"]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    input_ids = input_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    padding_length = MAX_LENGTH - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    labels += [-100] * padding_length

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor([1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids])
    }

# === Tokenized Dataset ===
class TokenizedDataset(Dataset):
    def __init__(self, raw_dataset):
        print("Tokenizing dataset...")
        self.data = [ex for ex in (tokenize_example(e) for e in raw_dataset) if ex is not None]
        print(f"Tokenization complete for {len(self.data)} samples.")

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

tokenized_train_dataset = TokenizedDataset(train_dataset)
tokenized_val_dataset = TokenizedDataset(val_dataset)

# === Label sanity check ===
label_lengths = [sum([1 for x in sample["labels"] if x != -100]) for sample in tokenized_train_dataset]
print("Avg target length:", sum(label_lengths) / len(label_lengths))
print("Any empty labels:", any(l == 0 for l in label_lengths))

# === Sanity Check ===
sample = tokenized_train_dataset[0]
decoded = tokenizer.decode([t for t in sample["labels"] if t != -100], skip_special_tokens=True)
print("Sample training target:", decoded)

# === Evaluation Function ===
rouge = load_metric("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}

# === Training arguments ===
print("Setting training arguments...")
training_args = TrainingArguments(
    output_dir="outputs_tinyllama",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-6,
    warmup_steps=100,
    max_grad_norm=1.0,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    report_to="none"
)

# === Define Trainer ===
print("Initializing trainer...")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# === Start Training ===
print("Starting training...")
trainer.train()
print("Training complete.")
