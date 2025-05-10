from huggingface_hub import hf_hub_download

try:
    hf_hub_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", filename="config.json")
    print("✅ Hugging Face connection is working and model repo exists!")
except Exception as e:
    print("❌ Error:", e)
