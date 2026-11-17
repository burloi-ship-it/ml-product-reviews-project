# 🛍️ Product Reviews Sentiment Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

An end-to-end machine learning project that classifies product reviews as **positive**, **neutral** or **negative**. It covers data cleaning, exploratory analysis, comparison of five classifiers, a reproducible training script and an interactive CLI for testing predictions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burloi-ship-it/ml-product-reviews-project/blob/main/notebook/product_reviews_analysis.ipynb)

---

## 📑 Table of Contents

- [🎯 Overview](#-overview)
- [🗂️ Dataset](#️-dataset)
- [🧹 Data Cleaning](#-data-cleaning)
- [🔬 Methodology](#-methodology)
- [📈 Results](#-results)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [▶️ Usage](#️-usage)
- [🔮 Possible Improvements](#-possible-improvements)
- [👤 Author](#-author)

---

## 🎯 Overview

| | |
|---|---|
| **Task** | Multi-class text classification (3 classes) |
| **Input** | Review title, review text, review length |
| **Output** | `positive` / `neutral` / `negative` |
| **Final model** | TF-IDF + Random Forest (scikit-learn `Pipeline`) |
| **Best test accuracy** | ~93% (Random Forest and Linear SVM) |

---

## 🗂️ Dataset

- 📄 File: `data/product_reviews_full.csv` (~38 MB)
- 🧾 **171,380 reviews**, 6 columns: `review_uuid`, `product_name`, `product_price`, `review_title`, `review_text`, `sentiment`
- 🔎 Raw data issues found during analysis:
  - 5,011 missing sentiment labels, 426 missing prices
  - Inconsistent label casing (`Positive` vs `positive`)
  - Prices stored as text, some with a `$` prefix

**Class distribution after cleaning (165,960 reviews):**

| Sentiment | Reviews | Share |
|-----------|--------:|------:|
| 😊 Positive | 107,619 | 64.8% |
| 😐 Neutral | 13,678 | 8.2% |
| 😡 Negative | 44,663 | 26.9% |

The dataset is **imbalanced**, with neutral as the smallest class.

---

## 🧹 Data Cleaning

- 🗑️ Dropped rows with missing values (171,380 → 165,960 rows)
- 🔤 Normalized sentiment labels (lowercase, trimmed) and converted to `category`
- 💲 Parsed `product_price` to `float` (removed currency symbols and non-numeric characters)
- 📏 Engineered a `review_length` feature (characters in `review_text`)
- ✂️ Dropped columns not used as features: `review_uuid`, `product_name`, `product_price`

---

## 🔬 Methodology

1. 📊 **Exploratory analysis**: missing-value heatmap, sentiment distribution, price and review length by sentiment (seaborn / matplotlib)
2. ✂️ **Split**: 80% train / 20% test, stratified by sentiment, `random_state=42` (132,768 train / 33,192 test)
3. 🔧 **Preprocessing** (`ColumnTransformer`):
   - `TfidfVectorizer` on `review_title`
   - `TfidfVectorizer` on `review_text`
   - `MinMaxScaler` on `review_length`
4. 🤖 **Models compared**: Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, Linear SVM
5. 💾 **Final model**: Random Forest pipeline trained on the full dataset and saved with `joblib`

---

## 📈 Results

Evaluation on the held-out 20% test set (33,192 reviews):

| Model | Accuracy | Macro F1 | Neutral F1 |
|-------|:--------:|:--------:|:----------:|
| 🌲 **Random Forest** | **0.93** | **0.83** | 0.62 |
| 📐 Linear SVM | 0.93 | 0.83 | 0.62 |
| 📉 Logistic Regression | 0.92 | 0.83 | 0.62 |
| 🌳 Decision Tree | 0.91 | 0.82 | 0.60 |
| 🧮 Naive Bayes | 0.90 | 0.75 | 0.42 |

**Key takeaways**

- ✅ Positive and negative reviews are classified reliably (F1 of 0.97 and 0.92 for Random Forest).
- ⚠️ The **neutral class is the weakest** (recall around 0.51–0.58 for most models), due to class imbalance and the ambiguity of neutral language.
- 🏆 Random Forest, Linear SVM and Logistic Regression perform almost identically, so the choice between them is driven by speed and model size rather than accuracy.
- ℹ️ The production model in `src/train_model.py` is trained on the **entire** dataset, so the metrics above come from the notebook's train/test split, not from the saved model.

---

## 📁 Project Structure

```text
.
├── data/
│   └── product_reviews_full.csv      # Dataset (~38 MB)
├── notebook/
│   └── product_reviews_analysis.ipynb  # EDA, cleaning, model comparison
├── src/
│   └── train_model.py                # Trains and saves the final pipeline
├── test/
│   └── model_testing.py              # Interactive CLI for predictions
├── .gitignore                        # Ignores *.pkl and /model
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- 🐍 Python 3.8+
- 🔧 Git

### Installation

```bash
git clone https://github.com/burloi-ship-it/ml-product-reviews-project.git
cd ml-product-reviews-project

python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Usage

**1. Train and save the model** (run from the project root):

```bash
mkdir model
python src/train_model.py
```

The pipeline is saved to `model/trained_model.pkl` (git-ignored). Training reads the dataset from the GitHub raw URL, so an internet connection is required.

**2. Test predictions interactively:**

```bash
python test/model_testing.py
```

```text
Model loaded successfully!
Type - exit - to exit prediction testing!
Enter review title: <your title>
Enter review text: <your review>
Predict sentiment: <positive | neutral | negative>
------------------------------
```

Type `exit` at any prompt to quit.

---

## 🔮 Possible Improvements

- [ ] ⚖️ Handle class imbalance (`class_weight="balanced"`, resampling) to improve neutral recall
- [ ] 🔍 Hyperparameter tuning with `GridSearchCV` / cross-validation
- [ ] 🧠 Try n-grams or transformer embeddings instead of plain TF-IDF
- [ ] 📦 Add a `requirements.txt` and unit tests with `pytest`
- [ ] 🌐 Serve the model through a small REST API (FastAPI)

---

## 👤 Author

**Burloi**

[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/burloi-ship-it)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/<your-profile>)

---

⭐ If you found this project useful, consider giving it a star!
