import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import numpy as np
import os

# Set title
st.title("Sentiment Analysis App")

# Load resources with caching
@st.cache_resource
def load_model_resources():
    model = load_model('model_files/simplified_lstm_20250612-172438_best.h5')
    with open('model_files/simplified_lstm_20250612-172438_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('model_files/simplified_lstm_20250612-172438_label_mapping.pickle', 'rb') as handle:
        label_mapping = pickle.load(handle)
    return model, tokenizer, label_mapping

model, tokenizer, label_mapping = load_model_resources()
reverse_mapping = {v: k for k, v in label_mapping.items()}

# Prediction function
def predict_sentiment(text):
    MAX_LEN = 20  # Should match training sequence length
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)
    return reverse_mapping[np.argmax(prediction)]

# Single text prediction
st.header("Single Text Analysis")
user_input = st.text_area("Enter text here:")
if st.button("Analyze Text"):
    if user_input:
        result = predict_sentiment(user_input)
        st.success(f"Predicted sentiment: **{result}**")
    else:
        st.warning("Please enter some text")

# CSV batch processing
st.header("Batch CSV Analysis")
uploaded_file = st.file_uploader("Upload CSV file:", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column")
    else:
        st.write("Preview:", df.head())
        
        if st.button("Process CSV"):
            df['prediction'] = df['text'].apply(predict_sentiment)
            st.dataframe(df)
            
            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name='sentiment_results.csv',
                mime='text/csv'
            )