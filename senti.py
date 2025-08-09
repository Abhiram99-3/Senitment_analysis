# sentiment_app.py

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = {
    "review": [
        "Congratulations! You've won a $500 Walmart gift card. Click here to claim now.",
        "Hey, are we still meeting for lunch today?",
        "URGENT: Your account has been suspended. Verify your details immediately.",
        "Don't forget about the meeting tomorrow at 10 AM.",
        "FREE entry in 2 a weekly competition! Text WIN to 80085 to claim your prize.",
        "Can you send me the report by EOD?",
        "Claim your free vacation now! Offer ends soon.",
        "Happy Birthday! Hope you have a great day.",
        "Limited time offer! Get 70% off all items today only.",
        "Let's catch up this weekend."
    ],
    "sentiment": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
print(df)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


accuracy = accuracy_score(y_test, model.predict(X_test_vec))


st.title("🎭 Sentiment Analysis App")

# Text input from user
user_input = st.text_area("Enter a movie review to analyze sentiment:")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.success(" ❌ 🤨😶😶🫥 ")
        else:
            st.error("✅ GOOD😊😊")
    else:
        st.warning("Please enter a review before clicking analyze.")













# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Sample dataset
# data = {
#     "review": [
#         "Congratulations! You've won a $500 Walmart gift card. Click here to claim now.",
#         "Hey, are we still meeting for lunch today?",
#         "URGENT: Your account has been suspended. Verify your details immediately.",
#         "Don't forget about the meeting tomorrow at 10 AM.",
#         "FREE entry in 2 a weekly competition! Text WIN to 80085 to claim your prize.",
#         "Can you send me the report by EOD?",
#         "Claim your free vacation now! Offer ends soon.",
#         "Happy Birthday! Hope you have a great day.",
#         "Limited time offer! Get 70% off all items today only.",
#         "I’m very disappointed with your service, never ordering again."
#     ],
#     "sentiment": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive/Spam, 0 = Negative/Not spam
# }

# df = pd.DataFrame(data)

# # Train-test split
# X_train_text, X_test_text, y_train, y_test = train_test_split(
#     df["review"], df["sentiment"], test_size=0.2, random_state=42
# )

# # TF-IDF Vectorization
# vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
# X_train_vec = vectorizer.fit_transform(X_train_text)
# X_test_vec = vectorizer.transform(X_test_text)

# # Model training
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_vec, y_train)

# # Accuracy
# accuracy = accuracy_score(y_test, model.predict(X_test_vec))
# print(f"Model Accuracy: {accuracy*100:.2f}%")

# # User input loop
# while True:
#     user_input = input("\nEnter text to analyze sentiment (or 'quit' to exit): ")
#     if user_input.lower() == "quit":
#         print("Exiting sentiment analysis.")
#         break
#     input_vec = vectorizer.transform([user_input])
#     prediction = model.predict(input_vec)[0]
    
#     if prediction == 1:
#         print("✅ Sentiment: Spam")
#     else:
#         print("❌ Sentiment: Not Spam")
