import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset (بدل ما نجيب داتا كبيرة)
data = {
    "text": [
        "I love this product", "Amazing experience", "Very good service",
        "I hate this", "Bad quality", "Worst ever",
        "It's okay", "Not bad", "Average experience"
    ],
    "label": ["positive", "positive", "positive",
              "negative", "negative", "negative",
              "neutral", "neutral", "neutral"]
}

df = pd.DataFrame(data)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Model
model = LogisticRegression()
model.fit(X, y)

def predict_sentiment(text):
    X_input = vectorizer.transform([text])
    return model.predict(X_input)[0]