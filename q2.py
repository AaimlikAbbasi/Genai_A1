import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from nltk.corpus import stopwords
import nltk
import pickle

# Download stopwords if not already present
nltk.download('stopwords')

# Define the file path for 'alllines.txt'
file_path = 'alllines.txt'

# Function to read lines from a file
def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return lines
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return []

# Preprocessing function to clean the text and remove stop words
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)    # Remove punctuation
    text = text.lower()        #convert into lower case 
    stop_words = set(stopwords.words('english'))  #load stops words
    words = text.split() # Tokenize the text into words
    filtered_words = [word for word in words if word not in stop_words] # Remove stop words
    return ' '.join(filtered_words)

# Read all lines from the file
lines = read_file(file_path)

# Tokenizer setup
tokenizer = Tokenizer()  # Initialize tokenizer
cleaned_texts = [preprocess_text(line) for line in lines] # Clean all lines
tokenizer.fit_on_texts(cleaned_texts)   # Fit the tokenizer on the cleaned text
sequences = tokenizer.texts_to_sequences(cleaned_texts) # Convert text to sequences of integers

# Create overlapping sequences of fixed length
def create_sequences(sequences, seq_length=5):
    X = []   # Input sequences
    y = []   # Target word (next word)
    for seq in sequences:
        for i in range(seq_length, len(seq)):
            X.append(seq[i-seq_length:i]) # Take previous 5 words
            y.append(seq[i]) # The next word is the target
    return X, y

# Generate sequences and targets (next word)
X, y = create_sequences(sequences, seq_length=5)

# Pad the sequences to ensure uniform input size
X_padded = pad_sequences(X, maxlen=5, padding='pre')
vocab_size = len(tokenizer.word_index) + 1
y_categorical = to_categorical(y, num_classes=vocab_size)

# Define the model architecture
model = Sequential()
embedding_size = 50
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=5))

model.add(LSTM(100, return_sequences=False))

model.add(Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_padded, y_categorical, epochs=35, batch_size=64, verbose=2)

# Save the model
model.save('text_prediction_model.h5')
print("Model saved as 'text_prediction_model.h5'")




# Save the tokenizer to a file
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)