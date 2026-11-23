
# ============================================================
# TF-IDF + Logistic Regression Baseline
# Digikala Reviews Project
# ============================================================

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. Configuration
# ============================================================

ID_COL = "id"
PRODUCT_COL = "product_id"
TEXT_COL = "text"
RATING_COL = "rate"

TARGETS = [
    "recommendation_status",
    "rating_band"
]

REQUIRED_COLUMNS = [
    ID_COL,
    PRODUCT_COL,
    TEXT_COL,
    RATING_COL,
    "recommendation_status"
]

RANDOM_STATE = 42


# ============================================================
# 2. Load datasets
# ============================================================

def load_data(train_path, validation_path):

    train_df = pd.read_csv(
        train_path,
        dtype=str,
        keep_default_na=False
    )

    val_df = pd.read_csv(
        validation_path,
        dtype=str,
        keep_default_na=False
    )

    # Check required columns
    for df, name in [
        (train_df, "Training"),
        (val_df, "Validation")
    ]:
        missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{name} dataset is missing columns: "
                f"{missing_columns}"
            )

    # Keep only required columns
    train_df = train_df[REQUIRED_COLUMNS].copy()
    val_df = val_df[REQUIRED_COLUMNS].copy()

    print("Training shape:", train_df.shape)
    print("Validation shape:", val_df.shape)

    return train_df, val_df


# ============================================================
# 3. Create rating_band
# ============================================================

def create_rating_band(df):

    df = df.copy()

    numeric_rate = pd.to_numeric(
        df[RATING_COL],
        errors="coerce"
    )

    df["rating_band"] = pd.NA

    df.loc[
        (numeric_rate >= 0) & (numeric_rate < 2.5),
        "rating_band"
    ] = "low"

    df.loc[
        (numeric_rate >= 2.5) & (numeric_rate < 4),
        "rating_band"
    ] = "medium"

    df.loc[
        (numeric_rate >= 4) & (numeric_rate <= 5),
        "rating_band"
    ] = "high"

    return df


# ============================================================
# 4. Prepare text data
# ============================================================

def prepare_data(df):

    df = df.copy()

    # Remove rows with missing or empty text
    df = df.dropna(subset=[TEXT_COL])

    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()

    df = df[df[TEXT_COL] != ""].copy()

    # Remove rows with missing or empty IDs
    for col in [ID_COL, PRODUCT_COL]:
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""].copy()

    # Convert rating to numeric
    df[RATING_COL] = pd.to_numeric(
        df[RATING_COL],
        errors="coerce"
    )

    return df


# ============================================================
# 5. Train and evaluate one model
# ============================================================

def train_tfidf_model(
    train_data,
    val_data,
    target_col,
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_iter=1000
):

    # Remove rows with missing target labels
    valid_train = train_data.dropna(
        subset=[target_col]
    ).copy()

    valid_val = val_data.dropna(
        subset=[target_col]
    ).copy()

    X_train = valid_train[TEXT_COL]
    y_train = valid_train[target_col]

    X_val = valid_val[TEXT_COL]
    y_val = valid_val[target_col]

    if len(valid_train) == 0:
        raise ValueError(
            f"No training samples for {target_col}."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            f"Training data for {target_col} "
            "contains fewer than two classes."
        )

    if len(valid_val) == 0:
        raise ValueError(
            f"No validation samples for {target_col}."
        )

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # Logistic Regression
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    # Validation prediction
    y_pred = model.predict(X_val_tfidf)

    # Evaluation
    metrics = {
        "target": target_col,
        "actual_train_size": len(valid_train),
        "validation_size": len(valid_val),

        "accuracy": accuracy_score(y_val, y_pred),

        "macro_f1": f1_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_val,
            y_pred
        )
    }

    report = classification_report(
        y_val,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_val,
        y_pred,
        labels=model.classes_
    )

    print(f"\nTarget: {target_col}")
    print(f"Training samples: {len(valid_train):,}")
    print(f"Validation samples: {len(valid_val):,}")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    return {
        "model": model,
        "tfidf": tfidf,
        "metrics": metrics,
        "report": report,
        "confusion_matrix": matrix
    }


# ============================================================
# 6. Train models for all targets
# ============================================================

def train_all_models(train_df, val_df):

    results = {}

    for target_col in TARGETS:

        print("\n" + "=" * 60)
        print(f"Training model for: {target_col}")
        print("=" * 60)

        result = train_tfidf_model(
            train_data=train_df,
            val_data=val_df,
            target_col=target_col
        )

        results[target_col] = result

    return results
