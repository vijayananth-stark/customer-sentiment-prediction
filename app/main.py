```python
from pathlib import Path

import joblib
import streamlit as st

from preprocessing import preprocess_text


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Sentiment Prediction",
    page_icon="😊",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("😊 Customer Sentiment Prediction")

st.write(
    "Enter a customer review below and the machine learning "
    "model will predict the sentiment."
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_models():

    model_path = MODEL_DIR / "sentiment_model.pkl"
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


try:

    model, vectorizer = load_models()

except Exception as e:

    st.error("Unable to load the ML model.")

    st.code(str(e))

    st.stop()


# =========================================================
# REVIEW INPUT
# =========================================================

review = st.text_area(
    "Enter Customer Review",
    placeholder="Example: The product quality is excellent and delivery was fast.",
    height=150
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("🔍 Predict Sentiment", use_container_width=True):

    if not review.strip():

        st.warning("Please enter a customer review.")

    else:

        try:

            # ---------------------------------------------
            # PREPROCESS TEXT
            # ---------------------------------------------

            processed_review = preprocess_text(review)

            # ---------------------------------------------
            # TF-IDF TRANSFORMATION
            # ---------------------------------------------

            review_vector = vectorizer.transform(
                [processed_review]
            )

            # ---------------------------------------------
            # PREDICTION
            # ---------------------------------------------

            prediction = model.predict(
                review_vector
            )[0]

            # ---------------------------------------------
            # PROBABILITY
            # ---------------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    review_vector
                )[0]

                classes = model.classes_

                confidence = {
                    str(class_name): round(
                        float(prob) * 100,
                        2
                    )
                    for class_name, prob in zip(
                        classes,
                        probabilities
                    )
                }

            else:

                confidence = {}

            # =================================================
            # DISPLAY RESULT
            # =================================================

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Sentiment: {prediction}"
            )

            # ---------------------------------------------
            # CONFIDENCE
            # ---------------------------------------------

            if confidence:

                st.subheader("Confidence")

                for class_name, probability in confidence.items():

                    st.write(
                        f"**{class_name}: {probability}%**"
                    )

                    st.progress(
                        min(int(probability), 100)
                    )

            # ---------------------------------------------
            # PROCESSED REVIEW
            # ---------------------------------------------

            with st.expander("View Processed Review"):

                st.write(processed_review)

        except Exception as e:

            st.error(
                "An error occurred during prediction."
            )

            st.code(str(e))
```
s
