🎭 Text Emotion Classifier
This repository contains an end-to-end Natural Language Processing (NLP) project that classifies text into six distinct emotions: Sadness, Anger, Love, Surprise, Fear, and Joy.

🚀 Project Overview
The goal of this project is to accurately predict the emotional state of a user based on their written input. The project covers the entire machine learning pipeline, from data preprocessing and feature engineering to model evaluation and deployment via a Streamlit web application.

📊 Dataset & Features
Source: The project utilizes a dataset of 16,000 text records.

Target Labels: Six emotional categories (encoded 0-5).

Pre-processing:

Lowercasing and punctuation removal.

Removal of numbers and non-ASCII characters (emojis).

Stopword removal using NLTK.
Model Performance
I experimented with multiple vectorization techniques and algorithms to find the best fit:
Model                      Vectorizer            Accuracy
Multinomial Naive Bayes  Bag of Words (BoW)        76.8%
Multinomial Naive Bayes  TF-IDF                    66.1%
Logistic Regression      TF-IDF                   86.28%

The final deployment uses the Logistic Regression model with TF-IDF Vectorization as it provided the highest accuracy.

💻 Tech Stack
Language: Python

Libraries: Pandas, Scikit-learn, NLTK, Matplotlib, Seaborn

Deployment: Streamlit
