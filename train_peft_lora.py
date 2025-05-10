
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from dataset import GODocDataset
from torch.utils.data import Dataset

# === Configuration ===
MODEL_NAME = "meta-llama/Llama-2-3b-hf"
TRAIN_TSV = "data/train_dataset.tsv"
VAL_TSV = "data/val_dataset.tsv"
OBO_PATH = "data/gene_ontology.obo"
MAX_LENGTH = 512

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# === Apply QLoRA ===
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)

# === Prepare datasets ===
train_dataset = GODocDataset(TRAIN_TSV, OBO_PATH)
val_dataset = GODocDataset(VAL_TSV, OBO_PATH)

def tokenize_example(example):
    input_text = "Summarize GO doc: " + example["input"]
    target_text = example["output"]
    model_inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    labels = tokenizer(
        target_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )["input_ids"]
    model_inputs["labels"] = labels
    return {k: torch.tensor(v) for k, v in model_inputs.items()}

class TokenizedDataset(Dataset):
    def __init__(self, raw_dataset):
        self.data = [tokenize_example(ex) for ex in raw_dataset]
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

tokenized_train_dataset = TokenizedDataset(train_dataset)
tokenized_val_dataset = TokenizedDataset(val_dataset)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir="outputs_llama3b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# === Define Trainer ===
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=MAX_LENGTH)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Start Training ===
trainer.train()
