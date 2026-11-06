import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re

BRANCH_COLORS = {
    "Disneyland California": "#38bdf8",
    "Disneyland Paris": "#f43f5e",
    "Disneyland HongKong": "#10b981"
}

SENTIMENT_COLORS = {
    "Positive": "#10b981",
    "Neutral": "#f59e0b",
    "Negative": "#f43f5e"
}

def load_disney_data(csv_path="data/disneyland_reviews.csv"):
    df = pd.read_csv(csv_path)
    return df

def train_sentiment_predictor(df):
    """Trains a fast TF-IDF + Logistic Regression rating predictor."""
    sample = df.sample(min(8000, len(df)), random_state=42)
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=4000, stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=300, random_state=42))
    ])
    
    pipeline.fit(sample["Review_Text"], sample["Rating"])
    return pipeline

def predict_single_review(text, pipeline):
    pred_rating = pipeline.predict([text])[0]
    probs = pipeline.predict_proba([text])[0]
    
    # Calculate lexical polarity
    pos_words = set(["great", "good", "amazing", "wonderful", "magical", "love", "loved", "best", "fantastic", "awesome", "fun", "happy", "excellent", "clean", "beautiful", "enjoyed", "favorite", "perfect", "fast"])
    neg_words = set(["bad", "terrible", "horrible", "worst", "dirty", "hate", "rude", "poor", "waste", "crowded", "expensive", "disappointed", "cold", "slow", "line", "lines", "broken", "wait", "long"])
    
    words = re.findall(r"\w+", str(text).lower())
    pos = sum(1 for w in words if w in pos_words)
    neg = sum(1 for w in words if w in neg_words)
    tot = pos + neg
    polarity = (pos - neg) / max(1, tot) if tot > 0 else 0.0
    sentiment = "Positive" if polarity > 0.15 else ("Negative" if polarity < -0.15 else "Neutral")
    
    return {
        "rating": pred_rating,
        "sentiment": sentiment,
        "polarity": round(polarity, 2),
        "word_count": len(words)
    }
