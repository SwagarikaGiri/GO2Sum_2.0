# T5 + Mixture of Experts (MoE) Training Script Template

import torch
from transformers import T5Tokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, T5Config
from torch.utils.data import Dataset
from dataset import GODocDataset
import torch.nn as nn
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Block

# === Custom T5 with Mixture of Experts ===
class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ff(x)

class MoEFFNLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)  # (batch, seq_len, num_experts)
        weights = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)

        output = 0
        for i in range(self.top_k):
            expert_idx = topk_indices[..., i]
            expert_weight = topk_weights[..., i].unsqueeze(-1)
            expert_out = torch.stack([self.experts[idx](x[b]) for b, idx in enumerate(expert_idx)], dim=0)
            output += expert_weight * expert_out

        return output

class T5MoEForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        for i, block in enumerate(self.encoder.block):
            block.layer[1].DenseReluDense = MoEFFNLayer(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_experts=4,
                top_k=2
            )
        for i, block in enumerate(self.decoder.block):
            block.layer[2].DenseReluDense = MoEFFNLayer(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_experts=4,
                top_k=2
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = T5Config.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        pretrained_model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
        return model

# === Configuration ===
MODEL_NAME = "t5-small"  # Use base model as backbone
TRAIN_TSV = "data/train_dataset_cleaned.tsv"
VAL_TSV = "data/val_dataset_cleaned.tsv"
OBO_PATH = "data/gene_ontology.obo"
MAX_LENGTH = 512

# === Load tokenizer ===
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# === Load MoE-enhanced T5 model ===
model = T5MoEForConditionalGeneration.from_pretrained(MODEL_NAME)

# === Prepare dataset ===
train_dataset = GODocDataset(TRAIN_TSV, OBO_PATH)
val_dataset = GODocDataset(VAL_TSV, OBO_PATH)

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

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

# === Tokenize datasets ===
tokenized_train_dataset = TokenizedDataset(train_dataset)
tokenized_val_dataset = TokenizedDataset(val_dataset)

# === Training setup ===
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
trainer.train()
