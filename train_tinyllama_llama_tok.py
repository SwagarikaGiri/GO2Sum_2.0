import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from dataset import GODocDataset
from torch.utils.data import Dataset

# === Configuration ===
print("📌 Configuration setup...")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Correct Hugging Face model repo
TRAIN_TSV = "data/train_dataset_cleaned.tsv"
VAL_TSV = "data/val_dataset_cleaned.tsv"
OBO_PATH = "data/gene_ontology.obo"
MAX_LENGTH = 512

# === Load tokenizer and model ===
print("📥 Loading tokenizer and model from Hugging Face...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
print("✅ Model and tokenizer loaded successfully.")

# === Prepare datasets ===
print("📁 Loading datasets...")
train_dataset = GODocDataset(TRAIN_TSV, OBO_PATH)
val_dataset = GODocDataset(VAL_TSV, OBO_PATH)
print(f"✅ Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

def tokenize_example(example):
    input_text = "Summarize GO doc: " + str(example.get("input", "")).strip()
    target_text = str(example.get("target", "")).strip()

    if not input_text or not target_text or target_text.lower() == "nan":
        return None

    model_inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    # Switch tokenizer to target mode
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )["input_ids"]

    # Replace PAD token ids with -100 so they’re ignored in loss calculation
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    model_inputs["labels"] = labels

    return {k: torch.tensor(v) for k, v in model_inputs.items()}

class TokenizedDataset(Dataset):
    def __init__(self, raw_dataset):
        print("🧪 Tokenizing dataset...")
        self.data = [ex for ex in (tokenize_example(e) for e in raw_dataset) if ex is not None]
        print(f"✅ Tokenization complete for {len(self.data)} samples.")
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

tokenized_train_dataset = TokenizedDataset(train_dataset)
tokenized_val_dataset = TokenizedDataset(val_dataset)

# 🧪 Optional: Sanity Check
sample = tokenized_train_dataset[0]
print("🧾 Input:", tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
print("🎯 Target:", tokenizer.decode([t for t in sample["labels"] if t != -100], skip_special_tokens=True))

# === Training arguments ===
print("⚙️ Setting training arguments...")
training_args = TrainingArguments(
    output_dir="outputs_tinyllama",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-6,          # ✅ Explicitly set low learning rate
    warmup_steps=100,            # ✅ Add warmup
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
print("🔧 Initializing trainer...")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=MAX_LENGTH,label_pad_token_id=-100)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Start Training ===
print("🚀 Starting training...")
trainer.train()
print("🏁 Training complete.")
