import transformers

print("✅ transformers version:", transformers.__version__)

# Test which imports work
try:
    from transformers import LlamaTokenizer
    print("✅ LlamaTokenizer import successful.")
except ImportError:
    print("❌ LlamaTokenizer import failed.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
    print("✅ Core HuggingFace imports successful.")
except ImportError as e:
    print("❌ Core transformers import failed:", e)
