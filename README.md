# 🏛️ NLP Sentiment Analysis - Colosseum Visitor Reviews

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/NLP-Sentiment%20Analysis-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/License-MIT-red?style=for-the-badge">
</p>

---

## 📖 Project Overview

This project presents a complete **Natural Language Processing (NLP)** pipeline for sentiment analysis using **Rome Colosseum Visitor Reviews (2019–2026)**.

The objective is to classify visitor reviews into **Positive**, **Neutral**, and **Negative** sentiments using both **Lexicon-Based** and **Machine Learning** approaches while addressing the challenge of **class imbalance**.

---

# 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Clean and preprocess textual reviews
- Apply VADER sentiment analysis
- Train Machine Learning classifiers
- Handle class imbalance
- Evaluate multiple models
- Perform error analysis
- Recommend the best-performing model

---

# 📂 Project Structure

```text
NLP-Sentiment-Analysis-Colosseum
│
├── plots/
│   ├── 1_rating_distribution.png
│   ├── 2_sentiment_comparison.png
│   ├── 3_trip_season.png
│   ├── 4_review_length.png
│   └── 5_model_comparison.png
│
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── lexicon_baseline.py
│   ├── ml_classifier.py
│   └── error_analysis.py
│
├── rome_colosseum_visitor_reviews_final.csv
├── run_pipeline.py
├── report.md
└── README.md
```

---

# ⚙️ Technologies Used

| Category | Tools |
|----------|-------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| NLP | NLTK, VADER |
| Machine Learning | Scikit-learn |
| Imbalance Handling | imbalanced-learn |

---

# 🔄 Project Workflow

```text
               Raw Dataset
                    │
                    ▼
           Data Loading & Audit
                    │
                    ▼
      Exploratory Data Analysis
                    │
                    ▼
        Text Preprocessing
                    │
                    ▼
         TF-IDF Vectorization
                    │
                    ▼
      Train / Test Split (80/20)
                    │
                    ▼
      Baseline ML Models
                    │
                    ▼
     Handle Class Imbalance
                    │
                    ▼
      Hyperparameter Tuning
                    │
                    ▼
       Model Evaluation
                    │
                    ▼
         Error Analysis
                    │
                    ▼
      Final Recommendation
```

---

# 🧹 Text Preprocessing

The preprocessing pipeline includes:

- ✅ Lowercasing
- ✅ URL Removal
- ✅ Punctuation Removal
- ✅ Number Removal
- ✅ Tokenization
- ✅ Stopword Removal
- ✅ Lemmatization
- ✅ Negation Preservation

### Example

**Original Review**

```text
The Colosseum was absolutely amazing, but the queues were terrible!
```

⬇️

**Processed Review**

```text
colosseum absolutely amazing queue terrible
```

---

# 📊 Exploratory Data Analysis

The project performs:

- Rating Distribution
- Sentiment Distribution
- Trip Type Analysis
- Travel Season Analysis
- Review Length Distribution
- Rating vs Sentiment Comparison

---

# 🤖 Machine Learning Models

### Baseline Models

- Logistic Regression
- Multinomial Naive Bayes

### Imbalance Handling Techniques

- Class Weighting
- Random Oversampling
- Random Undersampling
- GridSearchCV Hyperparameter Tuning

---

# 📈 Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- Macro F1-Score
- Balanced Accuracy
- Cohen's Kappa
- Confusion Matrix

> **Macro F1-score** was selected as the primary evaluation metric because the dataset is highly imbalanced.

---

# 📷 Output Visualizations

The project automatically generates:

- 📊 Rating Distribution
- 😊 Sentiment Distribution
- 🌍 Trip Type Analysis
- ☀️ Travel Season Analysis
- 📏 Review Length Distribution
- 📈 Model Comparison
- 📉 Confusion Matrix

---

# 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/SaadKhawer/NLP-Sentiment-Analysis-Colosseum.git
```

### Navigate

```bash
cd NLP-Sentiment-Analysis-Colosseum
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python run_pipeline.py
```

---

# 📌 Key Features

- ✔ End-to-End NLP Pipeline
- ✔ Professional EDA
- ✔ Text Cleaning
- ✔ TF-IDF Feature Engineering
- ✔ VADER Baseline
- ✔ Machine Learning Classification
- ✔ Class Imbalance Handling
- ✔ Hyperparameter Tuning
- ✔ Error Analysis
- ✔ Business Recommendation

---

# 💡 Future Improvements

- Fine-tune BERT
- Aspect-Based Sentiment Analysis
- Streamlit Dashboard
- Flask API Deployment
- Transformer Embeddings

---

# 👨‍💻 Author

**Saad Khawer**

BS Computer Science Student

AI | Machine Learning | NLP Enthusiast

---

## ⭐ If you found this project helpful, don't forget to star the repository!
