
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.preprocessing import normalize_persian_text
from app.model_loader import (
    load_transformer_model,
    load_tfidf_models,
    load_semi_supervised_model
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Load Transformer model
    transformer_model, tokenizer, transformer_metadata, device = load_transformer_model()

    app.state.transformer_model = transformer_model
    app.state.tokenizer = tokenizer
    app.state.transformer_metadata = transformer_metadata
    app.state.device = device

    # Load TF-IDF models
    tfidf_models, tfidf_metadata = load_tfidf_models()

    app.state.tfidf_models = tfidf_models
    app.state.tfidf_metadata = tfidf_metadata

        # Load Semi-Supervised model
    semi_model, semi_tokenizer, semi_metadata, semi_device = load_semi_supervised_model()

    app.state.semi_supervised_model = semi_model
    app.state.semi_supervised_tokenizer = semi_tokenizer
    app.state.semi_supervised_metadata = semi_metadata
    app.state.semi_supervised_device = semi_device

    yield


app = FastAPI(
    title="Digikala Reviews API",
    description="API for Persian product review analysis",
    version="1.0.0",
    lifespan=lifespan
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "message": "API is running"
    }


@app.get("/ready")
def ready():
    models_loaded = (
        hasattr(app.state, "transformer_model")
        and hasattr(app.state, "tfidf_models")
        and hasattr(app.state, "semi_supervised_model")
    )

    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="One or more models are not loaded."
        )

    return {
        "status": "ready",
        "models": {
            "tfidf": "loaded",
            "transformer": "loaded",
            "semi_supervised": "loaded"
        }
    }


@app.post("/normalize")
def normalize_text(request: TextRequest):

    normalized_text = normalize_persian_text(request.text)

    return {
        "original_text": request.text,
        "normalized_text": normalized_text
    }



@app.post("/tokenize")
def tokenize_text(request: TextRequest):

    # Normalize input text
    normalized_text = normalize_persian_text(request.text)

    if not normalized_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text is empty after normalization."
        )

    # Get tokenizer and maximum sequence length
    tokenizer = app.state.tokenizer
    max_length = app.state.transformer_metadata["max_length"]

    # Tokenize without truncation to check the original length
    full_encoding = tokenizer(
        normalized_text,
        add_special_tokens=True,
        truncation=False
    )

    original_token_count = len(full_encoding["input_ids"])

    # Tokenize with padding and truncation
    encoding = tokenizer(
        normalized_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Extract token IDs and attention mask
    input_ids = encoding["input_ids"][0].tolist()
    attention_mask = encoding["attention_mask"][0].tolist()

    # Convert token IDs to readable tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Count non-padding tokens
    token_count = sum(attention_mask)

    # Check whether truncation happened
    truncated = original_token_count > max_length

    return {
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_count": token_count,
        "truncated": truncated
    }



@app.post("/predict/transformer")
def predict_transformer(request: TextRequest):

    # Normalize input text
    normalized_text = normalize_persian_text(request.text)

    if not normalized_text.strip():
        raise HTTPException(
            status_code=400,
            detail="The review text is empty after normalization."
        )

    # Access loaded model components
    model = app.state.transformer_model
    tokenizer = app.state.tokenizer
    metadata = app.state.transformer_metadata
    device = app.state.device

    # Tokenize input text
    inputs = tokenizer(
        normalized_text,
        max_length=metadata["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Run model prediction
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        recommendation_probs = torch.softmax(
            outputs["recommendation_logits"],
            dim=1
        )[0]

        rating_probs = torch.softmax(
            outputs["rating_logits"],
            dim=1
        )[0]

    # Get class labels
    recommendation_labels = metadata["recommendation_labels"]
    rating_labels = metadata["rating_labels"]

    # Get recommendation prediction
    recommendation_index = int(
        torch.argmax(recommendation_probs).item()
    )

    recommendation_prediction = {
        "predicted_label": recommendation_labels[recommendation_index],
        "confidence": round(
            float(recommendation_probs[recommendation_index].item()), 4
        ),
        "probabilities": {
            label: round(float(probability.item()), 4)
            for label, probability in zip(
                recommendation_labels, recommendation_probs
            )
        }
    }

    # Get rating prediction
    rating_index = int(
        torch.argmax(rating_probs).item()
    )

    rating_prediction = {
        "predicted_label": rating_labels[rating_index],
        "confidence": round(
            float(rating_probs[rating_index].item()), 4
        ),
        "probabilities": {
            label: round(float(probability.item()), 4)
            for label, probability in zip(
                rating_labels, rating_probs
            )
        }
    }

    # Return both predictions
    return {
        "model_name": metadata["model_type"],
        "normalized text" : normalized_text,
        "predictions": {
            "recommendation_status": recommendation_prediction,
            "rating_band": rating_prediction
        }
    }



@app.post("/predict/tfidf")
def predict_tfidf(request: TextRequest):

    # Normalize input text
    normalized_text = normalize_persian_text(request.text)

    if not normalized_text.strip():
        raise HTTPException(
            status_code=400,
            detail="The review text is empty after normalization."
        )

    # Access loaded TF-IDF models
    models = app.state.tfidf_models

    predictions = {}

    # Predict both targets
    for target_name, model_data in models.items():

        vectorizer = model_data["vectorizer"]
        classifier = model_data["classifier"]

        # Convert text into TF-IDF features
        text_vector = vectorizer.transform([normalized_text])

        # Predict class probabilities
        probabilities = classifier.predict_proba(text_vector)[0]

        # Get class labels
        labels = classifier.classes_

        # Find predicted class
        predicted_index = int(probabilities.argmax())
        predicted_label = str(labels[predicted_index])

        # Prepare probability dictionary
        probability_dict = {
            str(label): round(float(probability), 4)
            for label, probability in zip(labels, probabilities)
        }

        predictions[target_name] = {
            "predicted_label": predicted_label,
            "confidence": round(float(probabilities[predicted_index]), 4),
            "probabilities": probability_dict
        }

    return {
        "model_name": "TF-IDF + Logistic Regression",
        "normalized text" : normalized_text,
        "predictions": predictions
    }



@app.post("/predict/semi-supervised")
def predict_semi_supervised(request: TextRequest):

    # Normalize input text
    normalized_text = normalize_persian_text(request.text)

    if not normalized_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text is empty after normalization."
        )

    # Get model components
    model = app.state.semi_supervised_model
    tokenizer = app.state.semi_supervised_tokenizer
    metadata = app.state.semi_supervised_metadata
    device = app.state.semi_supervised_device

    # Tokenize input text
    inputs = tokenizer(
        normalized_text,
        max_length=metadata["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move inputs to the model device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Make prediction
    with torch.inference_mode():

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        logits = outputs["recommendation_logits"]

        probabilities = torch.softmax(logits, dim=1)[0]

        confidence, predicted_class = torch.max(
            probabilities,
            dim=0
        )

    # Convert predicted class to label
    labels = metadata["recommendation_labels"]

    predicted_label = labels[predicted_class.item()]

    # Prepare probability dictionary
    probability_dict = {
        label: round(float(probability.item()), 4)
        for label, probability in zip(labels, probabilities)
    }

    return {
        "model": "semi_supervised",
        "normalized text" : normalized_text,
        "predictions": {
            "recommendation_status": {
                "predicted_label": predicted_label,
                "confidence": round(confidence.item(), 4),
                "probabilities": probability_dict
            }
        }
    }


@app.post("/predict/compare")
def compare_predictions(request: TextRequest):

    # Get predictions from all three models
    tfidf_result = predict_tfidf(request)

    transformer_result = predict_transformer(request)

    semi_supervised_result = predict_semi_supervised(request)

    # Return all predictions together
    return {
        "text": request.text,
        "comparison": {
            "tfidf": tfidf_result["predictions"],
            "transformer": transformer_result["predictions"],
            "semi_supervised": semi_supervised_result["predictions"]
        }
    }