📬 Spam & Sentiment Detection App

One-Liner: An end-to-end machine learning project to detect spam or sentiment in text using TF-IDF + Logistic Regression, with both terminal and Streamlit interfaces.

🚀 Overview

This project showcases the complete ML pipeline: from raw text data to feature engineering, model training, evaluation, and interactive prediction. It's perfect for demonstrating NLP basics in interviews or as a portfolio project.

✨ Features

🗂 Train a spam/sentiment classifier on a labeled dataset.

🧠 TF-IDF Vectorization for text feature extraction.

📊 Logistic Regression baseline model (easily replaceable with advanced models).

💻 Terminal Inference for quick testing.

🌐 Streamlit App for beautiful, interactive web predictions.

📈 Supports metrics like Accuracy, Precision, Recall, F1-score.

📂 Suggested Project Structure

spam-sentiment-app/
├─ data/                  # Datasets (CSV files)
├─ notebooks/             # EDA and training notebooks
├─ models/                # Saved model + vectorizer (pickle)
├─ app/                   # Streamlit app files
│  └─ sentiment_app.py
├─ sentiment_terminal.py  # Terminal inference script
├─ requirements.txt
└─ README.md

📊 Dataset

Use a larger dataset for better results — e.g., the SMS Spam Collection Dataset. Expected columns:

text → Message content

label → 1 for spam/positive, 0 for ham/negative

⚙️ Setup

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate    # windows

# Install dependencies
pip install -r requirements.txt

requirements.txt example:

pandas
scikit-learn
streamlit
textblob

🏋️ Training

Load dataset.

Clean text (lowercase, remove punctuation, stopwords).

Train-test split.

TF-IDF vectorize.

Train Logistic Regression.

Evaluate & save model/vectorizer in models/.

🖥 Terminal Mode

Run:

python sentiment_terminal.py

Type a sentence → get prediction (Spam/Not Spam). Type quit to exit.

🌍 Streamlit App

streamlit run app/sentiment_app.py

Shows model accuracy and live sentiment/spam predictions.

☁️ Deployment on Streamlit Cloud

Push repo to GitHub.

Create new Streamlit app → connect GitHub repo.

Set command: streamlit run app/sentiment_app.py.

Share public app link.

🔮 Improvements

Add bigger dataset & preprocessing.

Try Naive Bayes, SVM, RandomForest, or fine-tune BERT.

Add ROC-AUC, confusion matrix.

Integrate with Flask/FastAPI for API access.

💬 Sample Test Messages

Spam: Win a FREE iPhone 15 Pro! Claim now at scamlink.comHam: Hey, are we still meeting at 6 PM tonight?w: www.scamlink.com.

Ham example: Hey, are we still meeting for lunch today?
