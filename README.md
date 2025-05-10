# GO2Sum 2.0

**GO2Sum 2.0** is a deep learning framework for generating concise, human-readable summaries of protein functions using Gene Ontology (GO) terms. This version builds upon the original GO2Sum by integrating transformer models like **T5**, **TinyLlama**, and a custom **T5-MoE (Mixture of Experts)** model to improve generalization and interpretability.

## 🚀 Features

- Summarization of GO document descriptions for protein function prediction.
- Multi-model support: `T5`, `TinyLlama`, and MoE-enhanced variants.
- Clean, modular training scripts for each architecture.
- Preprocessing utilities for GO term documents.
- Support for `ROUGE`-based evaluation and dataset filtering.

## 📁 Project Structure

```
GO2Sum_2.0/
├── data/                   # Training, validation data
├── preprocessing/          # Scripts for preprocessing inputs
├── dataset.py              # GODocDataset class for input handling
├── train_t5.py             # Standard T5 training
├── train_t5_moe.py         # T5 with Mixture of Experts
├── train_tinyllama.py      # TinyLlama training script
├── generate_go_doc.py      # Generate GO document from GO term list
├── requirements.txt        # Python dependencies
├── check_conda_versions.sh # Utility to verify conda versions
├── README.md               # This file
└── ...
```

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SwagarikaGiri/GO2Sum_2.0.git
   cd GO2Sum_2.0
   ```

2. **Create and activate Conda environment**:
   ```bash
   conda create -n go2sum_env python=3.10
   conda activate go2sum_env
   ```

3. **Install required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

## 🧪 Dataset Format

Required files under `data/`:

- `train_dataset_cleaned.tsv`
- `val_dataset_cleaned.tsv`
- `gene_ontology.obo`

Each row in TSV must contain:
- `input` — GO term list
- `target` — Human-readable functional summary

## 🏋️‍♀️ Training

- **T5 model**:
  ```bash
  python train_t5.py
  ```

- **T5 with Mixture of Experts**:
  ```bash
  python train_t5_moe.py
  ```

- **TinyLlama model**:
  ```bash
  python train_tinyllama.py
  ```

## 📝 Generating GO Documents

Generate textual descriptions from GO term list using:
```bash
python generate_go_doc.py
```

## 📊 Evaluation

You can compute ROUGE scores using:
```bash
python train_casual_rouge.py
```

## 🔧 Environment Check

Verify conda environment versions:
```bash
bash check_conda_versions.sh
```

## 🤝 Contributing

Contributions and feature suggestions are welcome!

1. Fork the repository
2. Create a new branch: `git checkout -b feature/my-feature`
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👩‍🔬 Maintainer

**Swagarika Giri**  
PhD Candidate, Purdue University  
🔗 [GitHub](https://github.com/SwagarikaGiri)
