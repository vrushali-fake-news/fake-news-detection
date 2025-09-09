import streamlit as st
import joblib

# Title
st.title("📰 Fake News Detection")

# Input from user
news_text = st.text_area("Enter news content here:")

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Predict
if st.button("Predict"):
    if news_text.strip() != "":
        vector_input = vectorizer.transform([news_text])
        prediction = model.predict(vector_input)

        if prediction[0] == 1:
            st.success("✅ This news is Real.")
        else:
            st.error("🚨 This news is Fake!")
    else:
        st.warning("Please enter some text.")
