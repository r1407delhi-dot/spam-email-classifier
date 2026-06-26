"""
Spam Email Classifier — Task 03
Uses SMS Spam Collection dataset with TF-IDF + multiple models
"""

import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
print("=" * 60)
print("  SPAM EMAIL CLASSIFIER — Task 03")
print("=" * 60)

df = pd.read_csv('/home/claude/spam.tsv', sep='\t', header=None, names=['label', 'message'])
print(f"\n[DATA] Loaded {len(df)} messages")
print(f"  Ham  (non-spam): {(df['label'] == 'ham').sum()}")
print(f"  Spam           : {(df['label'] == 'spam').sum()}")

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Tokenization + stopword removal + lowercasing + punctuation removal"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)                    # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()                              # tokenization
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]  # stopword removal
    return ' '.join(tokens)

print("\n[PREPROCESSING] Tokenization + Stopword Removal + Lowercasing...")
df['clean_message'] = df['message'].apply(preprocess)
print(f"  Sample original : {df['message'].iloc[4][:60]}...")
print(f"  Sample cleaned  : {df['clean_message'].iloc[4][:60]}...")

# ─────────────────────────────────────────
# 3. TF-IDF VECTORIZATION
# ─────────────────────────────────────────
print("\n[TF-IDF] Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X = vectorizer.fit_transform(df['clean_message'])
y = (df['label'] == 'spam').astype(int)  # 1 = spam, 0 = ham

print(f"  Feature matrix shape: {X.shape}")
print(f"  Vocabulary size     : {len(vectorizer.vocabulary_)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 4. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────
models = {
    'Naive Bayes       ': MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
    'SVM (LinearSVC)   ': LinearSVC(C=1.0, max_iter=2000),
}

results = {}
print("\n[TRAINING & EVALUATION]")
print("-" * 60)
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 60)

best_f1 = 0
best_model = None
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results[name.strip()] = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1
    }

    print(f"  {name}  {acc:.4f}    {prec:.4f}    {rec:.4f}  {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name.strip()

print("-" * 60)
print(f"\n[BEST MODEL] {best_name} (F1 = {best_f1:.4f})")

# ─────────────────────────────────────────
# 5. DETAILED REPORT FOR BEST MODEL
# ─────────────────────────────────────────
y_pred_best = best_model.predict(X_test)
print(f"\n[CLASSIFICATION REPORT — {best_name}]")
print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

cm = confusion_matrix(y_test, y_pred_best)
print(f"[CONFUSION MATRIX]")
print(f"           Predicted Ham  Predicted Spam")
print(f"Actual Ham       {cm[0][0]:>5}           {cm[0][1]:>5}")
print(f"Actual Spam      {cm[1][0]:>5}           {cm[1][1]:>5}")

# ─────────────────────────────────────────
# 6. SAVE MODEL + VECTORIZER
# ─────────────────────────────────────────
joblib.dump(best_model, '/home/claude/spam_model.pkl')
joblib.dump(vectorizer, '/home/claude/tfidf_vectorizer.pkl')
print(f"\n[SAVED] Model → spam_model.pkl | Vectorizer → tfidf_vectorizer.pkl")

# ─────────────────────────────────────────
# 7. PREDICT FUNCTION (for interface)
# ─────────────────────────────────────────
def predict_spam(message, model=best_model, vec=vectorizer):
    """Given a raw message string, returns (label, confidence)"""
    cleaned = preprocess(message)
    features = vec.transform([cleaned])
    pred = model.predict(features)[0]
    label = "SPAM 🚨" if pred == 1 else "HAM ✅"

    # Confidence (probability) — LinearSVC uses decision_function
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        conf = max(proba) * 100
    else:
        score = model.decision_function(features)[0]
        # Sigmoid approximation
        conf = 1 / (1 + np.exp(-abs(score))) * 100

    return label, conf

# ─────────────────────────────────────────
# 8. SIMPLE INTERACTIVE INTERFACE
# ─────────────────────────────────────────
test_messages = [
    "Congratulations! You've won a FREE iPhone. Click here to claim now!!!",
    "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "URGENT: Your bank account has been suspended. Call 1800-XXX now.",
    "Can you send me the notes from today's lecture?",
    "Win cash prizes! Limited time offer. Text WIN to 80888 now.",
]

print("\n" + "=" * 60)
print("  LIVE PREDICTION DEMO")
print("=" * 60)
for msg in test_messages:
    label, conf = predict_spam(msg)
    print(f"\n  Message : {msg[:55]}...")
    print(f"  Result  : {label}  (Confidence: {conf:.1f}%)")

print("\n" + "=" * 60)
print("  DONE! Model ready. Run predict_spam(your_text) to classify.")
print("=" * 60)
