# ML Conversational Chatbot

An academic, production-ready **Text Analysis & Machine Learning** web application for automated intent classification and customer support helpdesk assistance.

---

## 📌 Project Overview
The **ML Conversational Chatbot** is a general-purpose AI helpdesk assistant designed to classify user queries into distinct intent categories using supervised machine learning techniques. Built with **scikit-learn**, **TF-IDF vectorization**, **Logistic Regression**, and **Flask**, this chatbot processes natural language user inputs, predicts intent probabilities in real-time, enforces a safety confidence threshold, and returns appropriate helpdesk responses.

---

## 🚨 Problem Statement
Traditional rule-based chatbots rely on hardcoded keyword matching or strict regular expressions, making them brittle and unable to handle semantic variations in user queries. Conversely, large language models (LLMs) and RAG setups can be computationally expensive and rely on third-party cloud APIs. This project addresses the need for a fast, self-contained, light-weight, and accurate machine learning solution for intent classification running entirely on local CPU resources.

---

## 🎯 Objectives
- Build a supervised machine learning pipeline for text classification.
- Preprocess natural language text and extract unigram/bigram numerical features using `TfidfVectorizer`.
- Train and evaluate a `LogisticRegression` classifier across 15 distinct support intents.
- Provide a modern responsive web dashboard with live intent/confidence visual feedback.
- Integrate browser-native speech input using the Web Speech API (`SpeechRecognition`).
- Deploy a self-contained local REST API using Flask.

---

## ✨ Features
- **Supervised Intent Classification**: Predicts 15 unique helpdesk intents with real-time probability confidence scoring.
- **Subtle Live ML Analysis Panel**: Displays the predicted intent badge, confidence percentage meter, and classification status.
- **Safety Confidence Fallback**: Triggers a helpful rephrasing fallback when the model confidence is below 30%.
- **Speech-to-Text Input**: Browser-native Web Speech API support allows voice queries without paid external APIs.
- **Interactive Dashboard**: Modern glassmorphism UI with typing indicators, timestamped message bubbles, suggested question chips, and a Clear Chat button.
- **Zero Third-Party LLM Dependencies**: Completely offline execution after installing basic Python packages.

---

## 🔬 Machine Learning Methodology

The machine learning workflow follows standard Natural Language Processing (NLP) text classification principles:

```
[ User Input ]
      │
      ▼
[ Text Preprocessing ] ──► (Lowercasing, Punctuation Removal, Whitespace Normalization)
      │
      ▼
[ TF-IDF Vectorizer ]  ──► (Extracts Unigram & Bigram Features, N-gram Range (1,2))
      │
      ▼
[ Logistic Regression ]──► (Calculates Probability Distribution Across 15 Intent Classes)
      │
      ▼
[ Confidence Threshold ]──► (>= 30%: Return Intent Response | < 30%: Trigger Fallback)
```

1. **Dataset**: `data/intents.json` contains 15 intent tags with multiple training phrases per intent.
2. **Preprocessing**: Converts text to lowercase, strips non-alphanumeric characters, and normalizes spacing.
3. **Feature Extraction**: `TfidfVectorizer(ngram_range=(1, 2))` transforms raw text into numerical TF-IDF feature vectors.
4. **Model**: `LogisticRegression(C=1.0, max_iter=1000)` calculates exact class probabilities via `predict_proba()`.
5. **Persistence**: The trained model and vectorizer are serialized into `models/chatbot_model.pkl` and `models/tfidf_vectorizer.pkl` using `joblib`.

---

## 🛠️ Technologies Used
- **Python 3.9+**
- **Flask**: Lightweight web framework for local REST endpoints.
- **scikit-learn**: Machine learning model training (`LogisticRegression`) and feature extraction (`TfidfVectorizer`).
- **joblib**: Model serialization and disk persistence.
- **NumPy**: Numerical operations and probability index lookup.
- **HTML5 / CSS3 / Vanilla JavaScript**: Responsive frontend UI dashboard.
- **Web Speech API**: Browser-native voice recognition.

---

## 📁 Project Structure

```
ML-Conversational-Chatbot/
│
├── app.py                  # Flask web server & REST API endpoint
├── train_model.py          # Machine learning model training script
├── chatbot.py              # ML classification engine & inference logic
├── requirements.txt        # Minimal Python dependencies
├── README.md               # Project documentation
│
├── data/
│   └── intents.json        # 15 support intents dataset with training examples
│
├── models/
│   ├── chatbot_model.pkl   # Serialized Logistic Regression classifier
│   └── tfidf_vectorizer.pkl# Serialized TF-IDF Vectorizer
│
├── templates/
│   └── index.html          # Web application frontend HTML dashboard
│
└── static/
    ├── style.css           # Modern CSS layout & visual styling
    └── script.js            # Frontend JavaScript controller & Web Speech API integration
```

---

## 🚀 Installation & Execution

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Clone / Navigate to Directory
```bash
cd ML-Conversational-Chatbot
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
Train the TF-IDF + Logistic Regression model and generate serialized `.pkl` model files:
```bash
python train_model.py
```

*Output:*
```text
============================================================
      ML CONVERSATIONAL CHATBOT - MODEL TRAINING           
============================================================
[*] Loaded 127 training samples across 15 unique intents.
[*] Vocabulary size: 431 features.
[*] Test Accuracy Score: 57.69%
[*] Retraining final model on full dataset for production...
[+] Model successfully saved to 'models/chatbot_model.pkl'
[+] Vectorizer successfully saved to 'models/tfidf_vectorizer.pkl'
============================================================
Training process complete!
```

### 5. Start the Web Server
Launch the Flask application:
```bash
python app.py
```

### 6. Access Application
Open your web browser and navigate to:
`http://127.0.0.1:5000`

---

## 💡 Example Inputs to Test

Try typing or speaking the following queries into the chatbot:

| User Query Example | Expected Intent Tag |
| :--- | :--- |
| *"Hello there"* | `greeting` |
| *"What can you help me with?"* | `capabilities` |
| *"How can I reset my password?"* | `password_reset` |
| *"How do I create a new account?"* | `account_creation` |
| *"How can I track my order shipment?"* | `order_tracking` |
| *"I want to cancel my order"* | `order_cancellation` |
| *"How do I request a refund?"* | `refund_request` |
| *"What are your working hours?"* | `working_hours` |
| *"How can I contact customer support?"* | `support_contact` |
| *"Payment failed during checkout"* | `payment_issue` |
| *"The website is crashing"* | `technical_issue` |
| *"Thank you so much"* | `thanks` |
| *"Goodbye!"* | `goodbye` |
| *"xyz random gibberish 1234"* | `low_confidence` (Triggers Safety Fallback) |

---

## 🔮 Future Enhancements
- Expand dataset with additional intent domain categories.
- Incorporate Lemmatization (NLTK or spaCy) during text preprocessing.
- Add multi-language intent classification using multilingual embeddings.
- Export chat session logs for analytics and continuous model retraining.

---

## 📄 License
Academic & Educational Use Only.
