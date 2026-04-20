import streamlit as st
from src.predict import predict_emotion

st.set_page_config(
    page_title="Text Emotion Recognition",
    page_icon="🧠",
    layout="centered"
)

st.title("Text Emotion Recognition System")
st.write("Enter transcribed speech text to predict its emotion.")

user_input = st.text_area(
    "Input Text",
    placeholder="Type your text here..."
)

if st.button("Predict Emotion"):
    if user_input.strip():
        predicted_label, confidence_score, message, top_predictions = predict_emotion(user_input)

        st.subheader("Prediction Result")
        st.success(f"Predicted Emotion: {predicted_label}")
        st.info(f"Confidence Score: {confidence_score:.4f}")

        st.subheader("Top Predictions")
        for label, score in top_predictions:
            st.write(f"- {label}: {score:.4f}")

        if message:
            st.warning(message)
    else:
        st.warning("Please enter some text before prediction.")