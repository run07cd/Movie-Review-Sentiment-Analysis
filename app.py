import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "sentiment_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

st.set_page_config(page_title="Movie Review Sentiment", page_icon="💬", layout="centered")
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify its sentiment as positive or negative.")

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.warning("The model has not been trained yet. Run python train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

review = st.text_area(
    "Movie review",
    placeholder="I enjoyed the story, the acting was strong, and the ending was satisfying.",
    height=180,
)

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.error("Enter a review first.")
    else:
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        classes = model.named_steps["classifier"].classes_
        probability_by_class = dict(zip(classes, probabilities))

        st.subheader(f"Prediction: {prediction.title()}")
        st.metric(
            "Model probability for this label",
            f"{probability_by_class[prediction]:.1%}",
        )
        st.caption("Probabilities are model estimates and may not be calibrated or correct for every review.")

st.divider()
st.subheader("Model evaluation")
st.caption("Results on a held-out test split from the NLTK movie_reviews corpus.")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
col2.metric("Macro precision", f"{metrics['precision_macro']:.1%}")
col3.metric("Macro F1", f"{metrics['f1_macro']:.1%}")

st.write(f"Training reviews: {metrics['training_reviews']} · Test reviews: {metrics['test_reviews']}")
st.write("Confusion matrix (rows = actual, columns = predicted):")
matrix = pd.DataFrame(
    metrics["confusion_matrix"],
    index=metrics["labels"],
    columns=metrics["labels"],
)
st.dataframe(matrix, use_container_width=True)
