/* ==========================================================================
   ML Conversational Chatbot - JavaScript Controller
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatWindow = document.getElementById('chatWindow');
    const welcomeState = document.getElementById('welcomeState');
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const micBtn = document.getElementById('micBtn');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const typingIndicator = document.getElementById('typingIndicator');
    const speechStatus = document.getElementById('speechStatus');

    // Live Analysis Elements
    const intentBadge = document.getElementById('intentBadge');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const statusAlert = document.getElementById('statusAlert');
    const statusText = document.getElementById('statusText');

    // Web Speech API Initialization
    let recognition = null;
    let isListening = false;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            isListening = true;
            micBtn.classList.add('listening');
            speechStatus.classList.remove('hidden');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (transcript) {
                userInput.value = transcript;
                userInput.focus();
            }
        };

        recognition.onerror = (event) => {
            console.warn('Speech recognition error:', event.error);
            stopSpeechRecognition();
        };

        recognition.onend = () => {
            stopSpeechRecognition();
        };
    } else {
        micBtn.title = 'Web Speech API is not supported in this browser';
        micBtn.style.opacity = '0.5';
    }

    function toggleSpeechRecognition() {
        if (!recognition) {
            alert('Web Speech API is not supported in your browser. Please use Chrome, Edge, or Safari.');
            return;
        }

        if (isListening) {
            recognition.stop();
        } else {
            try {
                recognition.start();
            } catch (err) {
                console.error('Failed to start speech recognition:', err);
            }
        }
    }

    function stopSpeechRecognition() {
        isListening = false;
        micBtn.classList.remove('listening');
        speechStatus.classList.add('hidden');
    }

    micBtn.addEventListener('click', toggleSpeechRecognition);

    // Form Submission & API Interaction
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        sendMessage(message);
    });

    async function sendMessage(messageText) {
        // Hide welcome state card on first message
        if (welcomeState && !welcomeState.classList.contains('hidden')) {
            welcomeState.classList.add('hidden');
        }

        // Add User Message to Chat Window
        appendMessage(messageText, 'user');
        userInput.value = '';

        // Show Typing Indicator
        showTyping(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: messageText })
            });

            if (!response.ok) {
                throw new Error(`Server returned error status ${response.status}`);
            }

            const data = await response.json();
            showTyping(false);

            // Append Bot Message
            appendMessage(data.response, 'bot', data);

            // Update ML Live Analysis Panel
            updateAnalysisPanel(data);

        } catch (error) {
            console.error('Error contacting backend API:', error);
            showTyping(false);
            appendMessage('Sorry, a connection error occurred. Please make sure the Flask server is running.', 'bot', {
                intent: 'error',
                confidence: 0.0,
                status: 'error'
            });
        }
    }

    // Helper: Append Message Bubble
    function appendMessage(text, sender, meta = null) {
        const row = document.createElement('div');
        row.classList.add('message-row', sender);

        const bubble = document.createElement('div');
        bubble.classList.add('message-bubble');
        bubble.textContent = text;

        const metaDiv = document.createElement('div');
        metaDiv.classList.add('message-meta');

        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        metaDiv.appendChild(document.createTextNode(timestamp));

        if (sender === 'bot' && meta) {
            const badge = document.createElement('span');
            badge.classList.add('intent-meta-badge');
            
            if (meta.status === 'fallback') {
                badge.classList.add('fallback-badge');
                badge.textContent = `Intent: ${meta.predicted_intent || meta.intent} (Low Conf)`;
            } else {
                badge.textContent = `Intent: ${meta.intent}`;
            }

            metaDiv.appendChild(document.createTextNode(' • '));
            metaDiv.appendChild(badge);
        }

        row.appendChild(bubble);
        row.appendChild(metaDiv);
        chatWindow.appendChild(row);

        // Scroll to bottom
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // Helper: Show/Hide Typing Indicator
    function showTyping(show) {
        if (show) {
            typingIndicator.classList.remove('hidden');
        } else {
            typingIndicator.classList.add('hidden');
        }
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // Helper: Update ML Live Analysis Panel
    function updateAnalysisPanel(data) {
        const intent = data.intent || 'unknown';
        const confidence = data.confidence ? (data.confidence * 100).toFixed(1) : '0.0';

        // Intent Badge
        intentBadge.textContent = intent;
        intentBadge.className = 'intent-tag';

        if (data.status === 'fallback') {
            intentBadge.classList.add('fallback');
            statusAlert.classList.remove('hidden');
            statusText.textContent = `Low Confidence (<30%). Safety fallback activated.`;
        } else {
            intentBadge.classList.add('active');
            statusAlert.classList.add('hidden');
        }

        // Confidence Meter
        confidenceValue.textContent = `${confidence}%`;
        confidenceBar.style.width = `${confidence}%`;
    }

    // Chip & Prompt Click Handlers
    document.addEventListener('click', (e) => {
        const chip = e.target.closest('.chip-btn, .prompt-chip');
        if (chip) {
            const query = chip.getAttribute('data-query');
            if (query) {
                sendMessage(query);
            }
        }
    });

    // Clear Chat History
    clearChatBtn.addEventListener('click', () => {
        // Clear all message rows
        const messageRows = chatWindow.querySelectorAll('.message-row');
        messageRows.forEach(row => row.remove());

        // Restore Welcome Card
        if (welcomeState) {
            welcomeState.classList.remove('hidden');
        }

        // Reset ML Live Analysis Panel
        intentBadge.textContent = 'None';
        intentBadge.className = 'intent-tag waiting';
        confidenceValue.textContent = '0.0%';
        confidenceBar.style.width = '0%';
        statusAlert.classList.add('hidden');
    });
});
