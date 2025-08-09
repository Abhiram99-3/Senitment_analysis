Spam & Sentiment Detection App

One-line: A simple end-to-end project to detect spam (and sentiment) from text using TF-IDF + Logistic Regression, with terminal and Streamlit interfaces.

Project Overview

A compact project demonstrating a full ML pipeline: data → preprocessing → TF-IDF vectorization → model training (Logistic Regression) → evaluation → inference. Includes a terminal script for quick testing and a Streamlit UI for easy demo and deployment.

Features

Train a spam/sentiment classifier on a dataset (example SMS/spam dataset).

TF-IDF feature extraction.

Logistic Regression model (easy to replace with other models).

Terminal-based inference (type text and get prediction).

Streamlit app for web demonstration and deployment.

Evaluation metrics: accuracy, precision, recall, F1 (can be added).

Repo Structure (suggested)

spam-sentiment-app/
├─ data/                  # datasets (csv)
├─ notebooks/             # EDA and training notebooks
├─ models/                # saved model and vectorizer (pickle)
├─ app/                   # streamlit app files
│  └─ sentiment_app.py
├─ sentiment_terminal.py  # terminal inference script
├─ requirements.txt
└─ README.md

Dataset

Use a real dataset for better results, e.g. the SMS Spam Collection Dataset (UCI) or any large labeled CSV with columns text and label (1 for spam, 0 for ham). Place it under data/.

Quick Setup

Create a virtual environment:

python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate    # windows

Install dependencies:

pip install -r requirements.txt

requirements.txt (example)

pandas
scikit-learn
streamlit
textblob

How to Train (example notebook)

Load dataset from data/.

Clean text (lowercase, remove punctuation, optional stopword removal).

Split: train_test_split.

Vectorize: TfidfVectorizer(stop_words='english', max_features=5000).

Train: LogisticRegression(max_iter=1000).

Evaluate: accuracy, precision, recall, F1-score.

Save model & vectorizer with pickle into models/.

Terminal Inference

Run the provided terminal script sentiment_terminal.py:

python sentiment_terminal.py

Type a message and see Spam or Not Spam prediction. Type quit to exit.

Streamlit App (app/sentiment_app.py)

To run locally:

streamlit run app/sentiment_app.py

The app shows model accuracy, and a text area for live predictions. If you deploy, upload models/ pickle files and the app will use them to predict.

Deployment (Streamlit Cloud)

Push the repo to GitHub (include requirements.txt).

Create a Streamlit Cloud app and connect your GitHub repo.

Provide a small startup command if required: streamlit run app/sentiment_app.py.

Add models/ folder or load model from an S3/Gist if file size limits apply.

Improvements & Next Steps

Use a larger labeled dataset (SMS Spam Collection, Enron emails, etc.).

Try other models: Naive Bayes, SVM, RandomForest, or fine-tune a transformer (BERT).

Add more metrics (confusion matrix, ROC-AUC) and cross-validation.

Add unit tests for preprocessing and model inference.

Build a REST API using FastAPI/Flask and add CI/CD pipeline.

Demo Test Messages

Spam example: Win a FREE iPhone 15 Pro! Click this link to claim your prize now: www.scamlink.com.

Ham example: Hey, are we still meeting for lunch today?