"""
AI Category Predictor
Uses Machine Learning (TF-IDF + Naive Bayes) to predict the category of an
expense from its title. Trained on training_data.csv.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score, StratifiedKFold

TRAINING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data.csv")


def build_model():
    return Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2))),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
        ])),
        ("classifier", MultinomialNB(alpha=0.1)),
    ])


def train_model():
    """Train the model on all data and return it."""
    data = pd.read_csv(TRAINING_FILE)
    model = build_model()
    model.fit(data["title"], data["category"])
    return model


def evaluate_model():
    """Return the 5-fold cross-validation accuracy of the model."""
    data = pd.read_csv(TRAINING_FILE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(build_model(), data["title"], data["category"], cv=cv)
    return scores.mean()


def predict_category(model, title):
    """Return (predicted_category, confidence) for an expense title."""
    probabilities = model.predict_proba([title])[0]
    best = probabilities.argmax()
    return model.classes_[best], float(probabilities[best])


if __name__ == "__main__":
    print(f"Model accuracy (5-fold cross-validation): {evaluate_model() * 100:.1f}%")
    model = train_model()
    for sample in ["Dominos pizza", "Auto to college", "Paracetamol tablets", "Jio recharge", "New t-shirt"]:
        category, confidence = predict_category(model, sample)
        print(f"{sample:<22} -> {category} ({confidence * 100:.0f}%)")
