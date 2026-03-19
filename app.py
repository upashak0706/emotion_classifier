import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the saved model and vectorizer
model = pickle.load(open('emotion_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Map numbers back to emotion names based on your unique_emotions order
# Update this list to match the exact order from your notebook's label encoding
emotions_map = {0: "Sadness", 1: "Anger", 2: "Love", 3: "Surprise", 4: "Fear", 5: "Joy"}

# Preprocessing function
def transform_text(text):
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    # Remove emojis/non-ascii
    text = ''.join([i for i in text if i.isascii()])
    # Remove stopwords
    words = text.split()
    text = " ".join([word for word in words if word not in stop_words])
    return text

# Streamlit UI
st.set_page_config(page_title="Emotion Classifier", page_icon="🎭")
st.title("🎭 Text Emotion Classifier")
st.write("Enter a sentence below to predict the underlying emotion.")

input_text = st.text_area("Enter the text here:", height=150)

if st.button('Predict Emotion'):
    if input_text.strip() != "":
        # 1. Preprocess
        transformed_sms = transform_text(input_text)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        prediction = emotions_map.get(result, "Unknown")
        
        st.header(f"Predicted Emotion: {prediction}")
    else:
        st.warning("Please enter some text first.")