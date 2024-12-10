from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import render_template
import numpy as np
import pickle
import re

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('text_prediction_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocessing function (same as used in training)
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text



# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')


# API route to get the next word prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data (text) from the POST request
    data = request.get_json()
    input_text = data['text']

    # Preprocess the input text
    input_seq = preprocess_text(input_text)
    input_seq = tokenizer.texts_to_sequences([input_seq])[0]
    input_seq = pad_sequences([input_seq], maxlen=5, padding='pre')

    # Make a prediction using the model
    pred = model.predict(input_seq, verbose=0)
    next_word_index = np.argmax(pred)
    
    # Get the predicted word from the tokenizer's index
    next_word = tokenizer.index_word.get(next_word_index, 'unknown')

    return jsonify({'next_word': next_word})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
