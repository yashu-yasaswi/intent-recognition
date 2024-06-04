import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tkinter as tk
from tkinter import ttk

# Set environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function to load dataset
def load_dataset(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, header=None, names=["intent", "query"])
    else:
        print(f"Error: Dataset file '{file_path}' not found.")
        exit()

# Load the ATIS Intents dataset
file_path = "atis_intents.csv"  # Update the path accordingly
atis_intents_data = load_dataset(file_path)

# Preprocess the dataset
queries = atis_intents_data["query"].tolist()
intents = atis_intents_data["intent"].tolist()

# Convert intents to numerical labels
label_encoder = LabelEncoder()
intents_encoded = label_encoder.fit_transform(intents)

# Split the dataset into training and validation sets
train_queries, val_queries, train_intents, val_intents = train_test_split(queries, intents_encoded, test_size=0.4, random_state=30)

# Define the maximum length of the sequences
max_len = 300  # Adjust based on your dataset's average query length

# Define function for preprocessing queries
def preprocess_queries(queries, max_len=300):
    tokenizer = Tokenizer()  # Use Tokenizer class for text preprocessing
    tokenizer.fit_on_texts(queries)
    sequences = tokenizer.texts_to_sequences(queries)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences, tokenizer

# Preprocess the queries
train_sequences, tokenizer = preprocess_queries(train_queries, max_len)
val_sequences, _ = preprocess_queries(val_queries, max_len)

# Define model architecture

def create_model(input_dim, output_dim, max_len):
    model = Sequential()
    
    # Embedding layer: maps input indices to dense vectors
    model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=max_len))
    
    # 1D Convolutional layer: applies 128 convolution filters of size 5
    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    
    # Bidirectional GRU layer: processes sequence in both directions
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    
    # Dropout layer: prevents overfitting by randomly setting 50% of inputs to 0
    model.add(Dropout(0.5))
    
    # Another Bidirectional GRU layer: returns only the last output
    model.add(Bidirectional(GRU(128)))
    
    # Fully connected output layer with softmax activation for classification
    model.add(Dense(output_dim, activation='softmax'))
    
    return model


# Custom Keras Classifier Wrapper
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, output_dim, max_len, learning_rate=0.004, epochs=10, batch_size=64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def fit(self, X, y):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X)
        sequences = self.tokenizer.texts_to_sequences(X)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        self.model = create_model(input_dim=self.input_dim, output_dim=self.output_dim, max_len=self.max_len)
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False)
        
        self.model.fit(
            padded_sequences, y,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        return self

    def predict(self, X):
        sequences = self.tokenizer.texts_to_sequences(X)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        predictions = self.model.predict(padded_sequences)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        sequences = self.tokenizer.texts_to_sequences(X)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        predictions = self.model.predict(padded_sequences)
        return predictions

# Use the custom Keras classifier
input_dim = len(tokenizer.word_index) + 1
output_dim = len(set(intents_encoded))

keras_classifier = KerasClassifier(input_dim=input_dim, output_dim=output_dim, max_len=max_len)

# Fit the model
keras_classifier.fit(train_queries, train_intents)

# Evaluate the model
val_pred = keras_classifier.predict(val_queries)
val_f1 = f1_score(val_intents, val_pred, average='weighted')
print(f"The best validation F1 score achieved: {val_f1}")

# Function to predict intent of a single query
def predict_intent(query, model, tokenizer, max_len):
    sequence = tokenizer.texts_to_sequences([query])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_intent_idx = np.argmax(prediction, axis=1)[0]
    predicted_intent = label_encoder.inverse_transform([predicted_intent_idx])[0]
    return predicted_intent

# Create the GUI interface
class IntentPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intent Prediction App")

        self.query_label = tk.Label(root, text="Enter your query:")
        self.query_label.pack(pady=5)

        self.query_entry = tk.Entry(root, width=50)
        self.query_entry.pack(pady=5)

        self.predict_button = tk.Button(root, text="Predict Intent", command=self.predict_intent)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=5)

    def predict_intent(self):
        query = self.query_entry.get()
        if query.lower() == 'exit':
            self.root.quit()
        else:
            predicted_intent = predict_intent(query, keras_classifier.model, keras_classifier.tokenizer, max_len)
            self.result_label.config(text=f"Predicted Intent: {predicted_intent}")

# Interactive query input and prediction with GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = IntentPredictionApp(root)
    root.mainloop()
