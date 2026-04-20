# Text Emotion Recognition

## Project Overview
This project is a Text-Based Emotion Recognition System developed using Python, NLP, Word2Vec, and RNN. It takes transcribed speech text as input and classifies it into one of the predefined emotion classes.

## Supported Emotions
- hate
- neutral
- anger
- love
- worry
- relief
- happiness
- fun
- empty
- enthusiasm
- sadness
- surprise
- boredom

## Features
- Text preprocessing and cleaning
- Tokenization
- Word2Vec embeddings
- RNN-based emotion classification
- 70/30 train-test split
- Performance evaluation using Precision, Recall, and F1-score
- Confusion matrix generation
- Confidence score display
- Streamlit UI for prediction
- Predict mode through command line

## Project Structure
```text
TEXT_EMOTION_RECOGNITION/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── report.py
│   ├── utils.py
│   └── visualization.py
│
├── main.py
├── ui.py
├── README.md
├── requirements.txt
└── project_report.md
