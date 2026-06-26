# 📧 Spam Email Classifier

A machine learning project that classifies messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and Scikit-learn.

> **Task 03** — Internship Project | Python · NLP · Machine Learning

---

## 🎯 Objective

Build a spam email classifier using Python and Scikit-learn that:
- Preprocesses raw text using tokenization, stopword removal, and TF-IDF vectorization
- Trains and compares multiple ML models (Naive Bayes, Logistic Regression, SVM)
- Evaluates performance using Accuracy, Precision, Recall, and F1-Score
- Provides a `predict_spam()` function to classify any new message

---

## 📁 Project Structure

```
spam-email-classifier/
│
├── spam_classifier.py    # Main script — preprocessing, training, evaluation
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── dataset/
    └── spam.tsv          # SMS Spam Collection dataset (5,572 messages)
```

---

## 📊 Dataset

- **Source:** SMS Spam Collection Dataset (UCI Machine Learning Repository)
- **Size:** 5,572 labeled messages
- **Classes:** Ham (4,825) | Spam (747)

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Scikit-learn | ML models + TF-IDF vectorizer |
| NLTK | Stopword removal |
| Pandas / NumPy | Data handling |
| Joblib | Model saving/loading |

---

## 🔄 Pipeline

```
Raw Text
   ↓
Preprocessing (lowercase → remove punctuation/numbers → stopword removal → tokenize)
   ↓
TF-IDF Vectorization (5000 features, unigrams + bigrams)
   ↓
Model Training (Naive Bayes / Logistic Regression / SVM)
   ↓
Evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
   ↓
predict_spam("your message here")
```

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 98.03% | 98.47% | 86.58% | 92.14% |
| Logistic Regression | 96.50% | 100.00% | 73.83% | 84.94% |
| **SVM (LinearSVC)** ✅ | **98.39%** | **97.12%** | **90.60%** | **93.75%** |

**Best Model: SVM (LinearSVC)** — highest F1-Score of **93.75%**

### Confusion Matrix (SVM)
```
                Predicted Ham   Predicted Spam
Actual Ham           962              4
Actual Spam           14            135
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/spam-email-classifier.git
cd spam-email-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the classifier**
```bash
python spam_classifier.py
```

**4. Predict a custom message**
```python
from spam_classifier import predict_spam

label, confidence = predict_spam("Congratulations! You won a FREE iPhone. Click now!")
print(label, confidence)
# Output: SPAM 🚨  (Confidence: 65.4%)
```

---

## 🧠 Key Concepts Covered

- **NLP Pipeline** — Tokenization, Stopword Removal, TF-IDF
- **Text Classification** — Naive Bayes, Logistic Regression, SVM
- **Model Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Model Persistence** — Saving/loading with Joblib

---

## 👤 Author

**Prem** — Electronics Engineering Student, GGSIPU Delhi  
GitHub: [@r1407delhi-dot](https://github.com/r1407delhi-dot)
