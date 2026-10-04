import os
from flask import Flask, render_template, request, jsonify
from chatbot import ChatbotEngine

app = Flask(__name__)

# Initialize Chatbot Engine
bot_engine = ChatbotEngine()


@app.route("/")
def index():
    """Renders the chatbot web interface."""
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Favicon endpoint to prevent 404 console logs."""
    return "", 204


@app.route("/predict", methods=["POST"])
@app.route("/analyze", methods=["POST"])
def predict():
    """Main classification API endpoint."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            data = request.form

        user_message = ""
        if data:
            user_message = data.get("message") or data.get("text") or data.get("query") or ""

        result = bot_engine.get_response(user_message)
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "intent": "system_error",
            "confidence": 0.0,
            "response": "An internal server error occurred while processing your request.",
            "status": "error"
        }), 500


@app.route("/api/intents", methods=["GET"])
def get_intents():
    """Returns available intent tags and count."""
    intents_list = list(bot_engine.intents.keys())
    return jsonify({
        "total_intents": len(intents_list),
        "intents": intents_list
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns sample questions for testing."""
    samples = [
        {"intent": "greeting", "sample": "Hello there"},
        {"intent": "capabilities", "sample": "What can you help me with?"},
        {"intent": "password_reset", "sample": "How can I reset my password?"},
        {"intent": "account_creation", "sample": "How do I create a new account?"},
        {"intent": "order_tracking", "sample": "How can I track my order?"},
        {"intent": "order_cancellation", "sample": "I want to cancel my order"},
        {"intent": "refund_request", "sample": "How can I get a refund?"},
        {"intent": "working_hours", "sample": "What are your working hours?"},
        {"intent": "support_contact", "sample": "How can I contact customer support?"}
    ]
    return jsonify({"samples": samples})


if __name__ == "__main__":
    print("Starting ML Conversational Chatbot Server...")
    # use_reloader=False prevents watchdog reloader from crashing background processes
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
