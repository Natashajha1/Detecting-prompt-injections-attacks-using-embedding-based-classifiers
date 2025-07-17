# 🛡️ Embedding-Based Classifiers for Prompt Injection Detection

This project presents a robust and modular framework for detecting **Prompt Injection Attacks** in **Large Language Models (LLMs)** using **semantic embeddings** and **traditional machine learning classifiers**. 
This system achieves strong classification performance using embeddings from **OpenAI**, **OctoAI**, and **MiniLM**, with downstream classifiers like **Random Forest** and **XGBoost**.

---

## 🚀 Features

- 🧠 Detects malicious prompts using static embeddings from LLMs
- 🧩 Supports three embedding sources:
  - `OpenAI text-embedding-3-small` (via API)
  - `OctoAI GTE-large` (via API)
  - `MiniLM` (local HuggingFace model)
- 🔎 Dimensionality reduction using PCA, t-SNE, and UMAP
- 📊 Training and evaluation using:
  - Logistic Regression
  - Random Forest
  - XGBoost
- 📁 467k+ prompt dataset balanced across benign/malicious labels
- 📈 Visualizations and saved metrics in reproducible format
- ✅ Modular, extensible architecture for plug-and-play embedding and classification

---

## 🏗️ Repository Structure

```bash
.
├── embedding.py                # Main script for generating embeddings (OpenAI, OctoAI, MiniLM)
├── create_test_split.py       # Creates stratified train-test split
├── download_openai_dataset.py # Merges and saves OpenAI embedding CSVs to pickle format
├── binary_classification.py   # Full ML training pipeline with metrics and CSV results
├── visualization.py           # Embedding visualizations (PCA, t-SNE, UMAP)
├── results/
│   ├── openai_random_forest.csv
│   ├── openai_xgb.csv
│   ├── openai_logistic_regression.csv
│   └── graphs/
├── Report.pdf 
└── README.md
```

## 💾 Dataset Preparation

> ⚠️ To keep this repository lightweight, the full dataset is **not included**.  
> You can generate it locally using **public Hugging Face datasets** with precomputed embeddings:

### 🔗 Pre-Generated Embedding Sources (CSV Format)

| Embedding Model             | Hugging Face Dataset |
|----------------------------|----------------------|
| **OpenAI (text-embedding-3-small)** | [📦 malicious-prompts-openai-embeddings](https://huggingface.co/datasets/ahsanayub/malicious-prompts-openai-embeddings) |
| **OctoAI (GTE-large)**             | [📦 malicious-prompts-octoai-embeddings](https://huggingface.co/datasets/ahsanayub/malicious-prompts-octoai-embeddings/) |
| **MiniLM (all-MiniLM-L6-v2)**      | [📦 malicious-prompts-minilm-embeddings](https://huggingface.co/datasets/ahsanayub/malicious-prompts-minilm-embeddings/) |

---

### 📌 Step 1: Download CSVs

Download all `.csv` files from your chosen embedding dataset and place them in the following folder:

```bash
embeddings/openai/     # For OpenAI embeddings
embeddings/octoai/     # For OctoAI embeddings
embeddings/minilm/     # For MiniLM embeddings
```

### Step 2: Merge and Convert to Pickle
Run:

```bash
python download_openai_dataset.py
```
### Step 3: Generate Train-Test Split
```bash
python create_test_split.py
```

### Step 4:Train Classifiers
```bash
python binary_classification.py
```
### Step 5: Visualize Embedding Clusters
```bash
python visualization.py
```

