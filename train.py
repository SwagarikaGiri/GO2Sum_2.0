# train.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from dataset import GODocDataset

# Load tokenizer and model
model_name = 'facebook/llama-7b'  # Replace with the appropriate LLaMA model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load dataset
dataset = GODocDataset(tsv_path='data/val_dataset.tsv', obo_path='data/go-basic.obo')

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example['input'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=True,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
