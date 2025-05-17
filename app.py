from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Model aur vectorizer load
try:
    train = pd.read_csv("https://drive.google.com/uc?export=download&id=1NrZEHpIiTwITIB10nNSmlae0SZcEJXPx", delimiter=",", quotechar='"', encoding='utf-8')
    print("Columns in dataset:", train.columns.tolist())  # Debugging ke liye
except Exception as e:
    print(f"Error loading dataset: {e}")
    train = pd.DataFrame()

# Model training
if not train.empty:
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(train['text'])
    y = train['sentiment']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
else:
    print("Dataset empty, model not trained.")
    model = None
    vectorizer = None

# Function to analyze topic-based sentiment
def analyze_topic(topic, data):
    if model is None or vectorizer is None:
        return None, None, 0
    # Filter tweets containing the topic
    topic_tweets = data[data['text'].str.contains(topic, case=False, na=False)]
    if len(topic_tweets) == 0:
        return None, None, 0
    
    # Predict sentiment for filtered tweets
    topic_vectors = vectorizer.transform(topic_tweets['text'])
    predictions = model.predict(topic_vectors)
    pred_probas = model.predict_proba(topic_vectors)
    
    # Calculate overall sentiment
    positive_count = sum(predictions)
    total = len(predictions)
    positive_percentage = (positive_count / total) * 100 if total > 0 else 0
    
    # Average confidence
    confidences = [max(proba) * 100 for proba in pred_probas]
    avg_confidence = round(np.mean(confidences), 2) if confidences else 0
    
    return positive_percentage, avg_confidence, total

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    tweet = None
    topic_result = None
    topic_confidence = None
    topic_count = 0
    topic = None

    if request.method == 'POST':
        if 'tweet' in request.form:
            # Single tweet analysis
            tweet = request.form['tweet']
            tweet_vector = vectorizer.transform([tweet])
            pred = model.predict(tweet_vector)
            pred_proba = model.predict_proba(tweet_vector)[0]
            prediction = "Positive" if pred[0] == 1 else "Negative"
            confidence = round(max(pred_proba) * 100, 2)

        elif 'topic' in request.form:
            # Topic-based analysis
            topic = request.form['topic']
            positive_percentage, topic_confidence, topic_count = analyze_topic(topic, train)
            if positive_percentage is not None:
                topic_result = f"{round(positive_percentage, 2)}% Positive"
            else:
                topic_result = "No tweets found for this topic"

    return render_template('index.html', 
                         prediction=prediction, 
                         confidence=confidence,
                         topic_result=topic_result,
                         topic_confidence=topic_confidence,
                         topic_count=topic_count,
                         topic=topic)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))