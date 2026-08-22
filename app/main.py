from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.preprocessing import preprocess_text


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"


# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------

app = FastAPI(
    title="Customer Sentiment Prediction API",
    description="NLP-based Customer Sentiment Prediction",
    version="1.0"
)


# ---------------------------------------------------------
# HTML TEMPLATES
# ---------------------------------------------------------

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model = joblib.load(
    MODEL_DIR / "sentiment_model.pkl"
)

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)


# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------

class ReviewRequest(BaseModel):
    review: str


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

@app.post("/predict")
async def predict_sentiment(request: ReviewRequest):

    processed_review = preprocess_text(
        request.review
    )

    review_vector = vectorizer.transform(
        [processed_review]
    )

    prediction = model.predict(
        review_vector
    )[0]

    probabilities = model.predict_proba(
        review_vector
    )[0]

    classes = model.classes_

    confidence = {
        class_name: round(float(prob) * 100, 2)
        for class_name, prob in zip(
            classes,
            probabilities
        )
    }

    return {
        "original_review": request.review,
        "processed_review": processed_review,
        "sentiment": prediction,
        "confidence": confidence
    }