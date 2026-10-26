import json
from pathlib import Path

import joblib
import nltk
from nltk.corpus import movie_reviews
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from sentiment_model import build_model


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "sentiment_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
LABELS = ["negative", "positive"]


def load_reviews():
    """Download the NLTK movie review corpus once, then read labeled reviews."""
    try:
        movie_reviews.fileids()
    except LookupError:
        print("Downloading the movie_reviews dataset...")
        if not nltk.download("movie_reviews"):
            raise RuntimeError("Dataset download failed. Check your internet connection.")

    file_ids = movie_reviews.fileids()
    texts = [movie_reviews.raw(file_id) for file_id in file_ids]
    labels = [
        "positive" if "pos" in movie_reviews.categories(file_id) else "negative"
        for file_id in file_ids
    ]
    return texts, labels


def main():
    texts, labels = load_reviews()
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = build_model()
    print(f"Training on {len(X_train)} reviews...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    report = classification_report(
        y_test, predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)

    metrics = {
        "dataset": "NLTK movie_reviews",
        "total_reviews": len(texts),
        "training_reviews": len(X_train),
        "test_reviews": len(X_test),
        "labels": LABELS,
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nEvaluation on held-out test reviews")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"Macro F1:  {f1:.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(matrix)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
