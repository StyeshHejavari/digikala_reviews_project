
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from app.model import MultiTaskParsBERT


# Define artifact directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "transformer"


def load_transformer_model():

    # Select the available device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load model metadata
    metadata_path = ARTIFACT_DIR / "metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Read model configuration from metadata
    model_name = metadata["model_name"]
    dropout_rate = metadata["dropout_rate"]

    num_recommendation_classes = len(
        metadata["recommendation_labels"]
    )

    num_rating_classes = len(
        metadata["rating_labels"]
    )

    # Create the model architecture
    model = MultiTaskParsBERT(
        model_name=model_name,
        num_recommendation_classes=num_recommendation_classes,
        num_rating_classes=num_rating_classes,
        dropout_rate=dropout_rate
    )

    # Load trained model weights
    model_path = ARTIFACT_DIR / "model_state_dict.pt"

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    # Move model to the selected device
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Load tokenizer from local files
    tokenizer_path = ARTIFACT_DIR / "tokenizer"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True
    )

    return model, tokenizer, metadata, device



import joblib

def load_tfidf_models():
    tfidf_dir = PROJECT_DIR / "artifacts" / "tfidf"

    metadata_path = tfidf_dir / "metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    models = {}

    for target_name, config in metadata["targets"].items():

        vectorizer_path = tfidf_dir / config["vectorizer_file"]
        model_path = tfidf_dir / config["model_file"]

        vectorizer = joblib.load(vectorizer_path)
        classifier = joblib.load(model_path)

        models[target_name] = {
            "vectorizer": vectorizer,
            "classifier": classifier
        }

    return models, metadata



# Semi-Supervised model artifact directory
SEMI_SUPERVISED_ARTIFACT_DIR = (
    PROJECT_DIR / "artifacts" / "semi_supervised"
)


def load_semi_supervised_model():

    # Select the available device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load model metadata
    metadata_path = SEMI_SUPERVISED_ARTIFACT_DIR / "metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Read model configuration
    model_name = metadata["model_name"]
    dropout_rate = metadata["dropout_rate"]
    num_classes = metadata["num_classes"]

    # Create the model architecture
    model = MultiTaskParsBERT(
        model_name=model_name,
        num_recommendation_classes=num_classes,
        num_rating_classes=num_classes,
        dropout_rate=dropout_rate
    )

    # Load trained model weights
    model_path = (
        SEMI_SUPERVISED_ARTIFACT_DIR / "model_state_dict.pt"
    )

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    # Move model to device and set evaluation mode
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = (
        SEMI_SUPERVISED_ARTIFACT_DIR / "tokenizer"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True
    )

    return model, tokenizer, metadata, device
