import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from dataset import GODocDataset
from torch.utils.data import Dataset

# === Configuration ===
print(" Configuration setup...")
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
    torch_dtype=torch.float16,
    trust_remote_code=True
)
print(" Model and tokenizer loaded successfully.")

# === Prepare datasets ===
print("Loading datasets...")
train_dataset = GODocDataset(TRAIN_TSV, OBO_PATH)
val_dataset = GODocDataset(VAL_TSV, OBO_PATH)
print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

# === Causal LM Tokenization ===
def tokenize_example(example):
    prompt = "Summarize GO doc: " + str(example.get("input", "")).strip()
    target = str(example.get("target", "")).strip()

    if not prompt or not target or target.lower() == "nan":
        return None

    full_input = prompt + "\n" + target
    tokenized = tokenizer(full_input, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    prompt_len = len(tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    tokenized["labels"] = labels
    return {k: torch.tensor(v) for k, v in tokenized.items()}

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

# === Sanity Check ===
sample = tokenized_train_dataset[0]
decoded = tokenizer.decode([t for t in sample["labels"] if t != -100], skip_special_tokens=True)
print(" Sample training target:", decoded)

# === Training arguments ===
print(" Setting training arguments...")
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
    data_collator=data_collator
)

# === Start Training ===
print(" Starting training...")
trainer.train()
print("Training complete.")
