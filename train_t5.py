import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from dataset import GODocDataset
from torch.utils.data import Dataset

# === Configuration ===
MODEL_NAME = "t5-small"  # Replace with your MoE-augmented T5 model if available
TRAIN_TSV = "data/train_dataset_cleaned.tsv"
VAL_TSV = "data/val_dataset_cleaned.tsv"
OBO_PATH = "data/gene_ontology.obo"
MAX_LENGTH = 512

# === Load tokenizer and model ===
print("📥 Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float32
)
print("✅ Model and tokenizer loaded.")

# === Prepare datasets ===
print("📁 Loading datasets...")
train_dataset = GODocDataset(TRAIN_TSV, OBO_PATH)
val_dataset = GODocDataset(VAL_TSV, OBO_PATH)

# === Tokenize examples ===
def tokenize_example(example):
    prompt = "summarize: " + str(example.get("input", "")).strip()
    target = str(example.get("target", "")).strip()

    if not prompt or not target or target.lower() == "nan":
        return None

    model_inputs = tokenizer(prompt, max_length=MAX_LENGTH, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, max_length=MAX_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return {k: torch.tensor(v) for k, v in model_inputs.items()}

class TokenizedDataset(Dataset):
    def __init__(self, raw_dataset):
        self.data = [ex for ex in (tokenize_example(e) for e in raw_dataset) if ex is not None]
        print(f"✅ Tokenized {len(self.data)} examples.")

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

tokenized_train_dataset = TokenizedDataset(train_dataset)
tokenized_val_dataset = TokenizedDataset(val_dataset)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir="outputs_t5_moe",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    report_to="none"
)

# === Trainer ===
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Train ===
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete.")
