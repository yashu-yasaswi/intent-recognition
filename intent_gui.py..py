import numpy as np
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GRU, TimeDistributed, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Define model parameters (these should match your training script)
n_classes = 17  # Set this to the number of classes in your dataset
n_vocab = 1000  # Set this to the size of your vocabulary + 1 (for padding)

# Load the trained model
model = Sequential()
model.add(Embedding(n_vocab, 100))
model.add(Conv1D(64, 5, padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(GRU(100, return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile('rmsprop', 'sparse_categorical_crossentropy')

# Provide the correct path to your model weights file
weights_path = 'C:\\ATIS.keras-master (1) (extract.me)\\ATIS.keras-master\\best_model_weights.h5'
model.load_weights(weights_path)

# Load the label encoder
label_encoder = LabelEncoder()
classes_path = 'C:\\ATIS.keras-master (1) (extract.me)\\ATIS.keras-master\\classes.npy'
label_encoder.classes_ = np.load(classes_path, allow_pickle=True)

# Load word index dictionary
w2idx_path = 'C:\\ATIS.keras-master (1) (extract.me)\\ATIS.keras-master\\w2idx.npy'
w2idx = np.load(w2idx_path, allow_pickle=True).item()

def preprocess_query(query):
    """Preprocess the query for prediction."""
    sequence = [w2idx.get(word, 0) for word in query.split()]
    return np.array(sequence).reshape(1, -1)

def predict_intent():
    query = query_entry.get()
    if not query:
        messagebox.showwarning("Input Error", "Please enter a query.")
        return

    sequence = preprocess_query(query)
    pred = model.predict(sequence)
    intent = label_encoder.inverse_transform([np.argmax(pred)])
    result_label.config(text=f"Predicted Intent: {intent[0]}")

# Create the Tkinter application
root = tk.Tk()
root.title("Intent Prediction")

tk.Label(root, text="Enter your query:").pack(pady=10)
query_entry = tk.Entry(root, width=50)
query_entry.pack(pady=10)

tk.Button(root, text="Predict", command=predict_intent).pack(pady=10)

result_label = tk.Label(root, text="Predicted Intent: ")
result_label.pack(pady=10)

root.mainloop()
