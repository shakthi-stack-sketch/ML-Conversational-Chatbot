import os
import json
import random
import re
import joblib
import numpy as np

MODEL_PATH = os.path.join("models", "chatbot_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
DATA_PATH = os.path.join("data", "intents.json")

# Confidence threshold calibrated for 15-class Logistic Regression
CONFIDENCE_THRESHOLD = 0.15


def preprocess_text(text: str) -> str:
    """Preprocess user text matching training normalization."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ChatbotEngine:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.intents = {}
        self.load_resources()

    def load_resources(self):
        """Loads trained model, vectorizer, and intent dictionary."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                "Trained model or vectorizer not found. Run 'train_model.py' first!"
            )

        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                for intent in data.get("intents", []):
                    self.intents[intent["tag"]] = intent.get("responses", [])

    def get_response(self, user_message: str) -> dict:
        """Classifies the user intent using trained TF-IDF + Logistic Regression."""
        if not user_message or not user_message.strip():
            return {
                "intent": "empty_input",
                "confidence": 0.0,
                "response": "Please enter a valid text message or use voice input.",
                "status": "warning"
            }

        cleaned_text = preprocess_text(user_message)
        if not cleaned_text:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "response": "I'm not confident that I understood that. Could you rephrase your question?",
                "status": "fallback"
            }

        # Vectorize text
        tfidf_features = self.vectorizer.transform([cleaned_text])

        # Get class probabilities
        probabilities = self.model.predict_proba(tfidf_features)[0]
        max_index = np.argmax(probabilities)
        confidence = float(probabilities[max_index])
        predicted_intent = self.model.classes_[max_index]

        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "intent": "low_confidence",
                "predicted_intent": predicted_intent,
                "confidence": round(confidence, 4),
                "response": "I'm not confident that I understood that. Could you rephrase your question?",
                "status": "fallback"
            }

        # Select response for predicted intent
        possible_responses = self.intents.get(
            predicted_intent,
            ["I understand your query, but I don't have a specific answer right now."]
        )
        selected_response = random.choice(possible_responses)

        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "response": selected_response,
            "status": "success"
        }
