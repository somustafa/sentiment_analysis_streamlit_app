import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


st.title("IMDB Sentiment Analysis 🎬")


model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


metrics = {"accuracy": 0.0, "f1": 0.0}
try:
    metrics = pickle.load(open("lr_metrics.pkl", "rb"))
except:
    pass


user_review = st.text_area("Write your review:")

if st.button("Analyze"):
    if user_review.strip() == "":
        st.warning("Please write a review first.")
    else:
        review = user_review.lower()
        review = re.sub(r"[^\w\s]", "", review)
        review = " ".join(w for w in review.split() if w not in stop_words)

        vect = vectorizer.transform([review])
        pred = model.predict(vect)[0]

        if pred == "positive":
            st.success(f"Prediction: {pred.capitalize()}")
        else:
            st.error(f"Prediction: {pred.capitalize()}")


st.subheader("Model Performance")
st.write("Accuracy:", round(metrics.get("accuracy", 0.0), 4))
st.write("F1 Score:", round(metrics.get("f1", 0.0), 4))

if st.checkbox("Show Confusion Matrix"):
    with open("test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    X_test_raw = test_data['X_test_raw']
    y_test = test_data['y_test']

    with open("y_pred_test.pkl", "rb") as f:
        y_pred_test = pickle.load(f)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)