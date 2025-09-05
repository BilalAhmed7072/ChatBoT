from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
print("✅ Starting Flask app setup...")
app = Flask(__name__)
CORS(app)

# Load model and tokenizer
print("✅ Loading model...")
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Model loaded successfully!")
conversation_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_json()
    input_text = data['prompt']

    # Build conversation history string
    history = "\n".join(conversation_history)

    # Tokenize input
    inputs = tokenizer.encode_plus(history + "\n" + input_text, return_tensors="pt")

    # Generate response
    outputs = model.generate(**inputs, max_length=60)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Update conversation history
    conversation_history.append(f"You: {input_text}")
    conversation_history.append(f"Bot: {response}")

    return jsonify({"response": response})

if __name__ == "__main__":
    print("✅ Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
