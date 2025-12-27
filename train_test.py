import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

data = pd.read_csv("IMDB Dataset.csv")

data['review'] = data['review'].str.lower()
data['review'] = data['review'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
data['review'] = data['review'].apply(
    lambda x: ' '.join(w for w in x.split() if w not in stop_words)
)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['review'])
y = data['sentiment']

# sadələşdirilmiş versiya
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test, pos_label="positive")
cm = confusion_matrix(y_test, y_pred_test)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

metrics = {"accuracy": accuracy, "f1": f1}
pickle.dump(metrics, open("lr_metrics.pkl", "wb"))

pickle.dump(y_pred_test.tolist(), open("y_pred_test.pkl", "wb"))
