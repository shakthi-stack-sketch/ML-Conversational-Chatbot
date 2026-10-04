import os
import json
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join("data", "intents.json")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


def preprocess_text(text: str) -> str:
    """Preprocess input text: lowercasing, removing non-alphanumeric characters, and extra spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(filepath: str):
    """Load intents dataset from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    labels = []

    for intent in data.get("intents", []):
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            cleaned_pattern = preprocess_text(pattern)
            if cleaned_pattern:
                sentences.append(cleaned_pattern)
                labels.append(tag)

    return sentences, labels


def train():
    print("=" * 60)
    print("      ML CONVERSATIONAL CHATBOT - MODEL TRAINING           ")
    print("=" * 60)

    # 1. Load Data
    sentences, labels = load_dataset(DATA_PATH)
    print(f"[*] Loaded {len(sentences)} training samples across {len(set(labels))} unique intents.")

    # 2. Train / Test Split for Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 3. Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"[*] Vocabulary size: {len(vectorizer.get_feature_names_out())} features.")

    # 4. Model Training (Logistic Regression with C=10.0 for sharper probabilities)
    classifier = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    classifier.fit(X_train_tfidf, y_train)

    # 5. Evaluation
    y_pred = classifier.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"[*] Test Accuracy Score: {acc * 100:.2f}%")
    print("\n[ Classification Report ]\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 6. Fit on FULL dataset for production deployment
    print("[*] Retraining final model on full dataset for production...")
    full_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_full_tfidf = full_vectorizer.fit_transform(sentences)

    full_classifier = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    full_classifier.fit(X_full_tfidf, labels)

    # 7. Save Model and Vectorizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(full_classifier, MODEL_PATH)
    joblib.dump(full_vectorizer, VECTORIZER_PATH)

    print(f"[+] Model successfully saved to '{MODEL_PATH}'")
    print(f"[+] Vectorizer successfully saved to '{VECTORIZER_PATH}'")
    print("=" * 60)
    print("Training process complete!")


if __name__ == "__main__":
    train()
