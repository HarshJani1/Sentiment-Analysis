import re
import pickle
import nltk
import streamlit as st
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the trained model and vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Initialize PorterStemmer and stopwords
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
if 'not' in all_stopwords:
    all_stopwords.remove('not')  # Keep 'not' for negation

# Function to preprocess input text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in all_stopwords]
    return ' '.join(text)

# Function to predict sentiment
def predict_user_input(text):
    processed_text = preprocess_text(text)
    transformed_text = loaded_vectorizer.transform([processed_text]).toarray()
    prediction = loaded_model.predict(transformed_text)
    return prediction[0]

# Streamlit App
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🔍")
st.title("🔍 Sentiment Analysis App")
st.write("Enter your review below and find out the sentiment!")

user_input = st.text_area("Enter a review:", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment = predict_user_input(user_input)
        sentiment_label = "Positive 😊" if sentiment == 1 else "Negative 😥"
        st.success(f"Predicted Sentiment: **{sentiment_label}**")
