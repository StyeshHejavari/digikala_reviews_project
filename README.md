# Digikala Reviews: Persian Text Classification

A machine learning project for analyzing Persian customer reviews from Digikala using classical machine learning and Transformer-based models.

The project focuses on predicting two aspects of a product review using only its text:

1. **Recommendation Status:** Whether the customer recommends the product.
2. **Rating Band:** Whether the review corresponds to a low, medium, or high rating.

The project also explores whether semi-supervised learning with pseudo-labeling can improve the performance of the fine-tuned Transformer model.

## Project Objectives

- Analyze and understand a large Persian review dataset.
- Normalize Persian text while preserving sentiment and negation.
- Prevent data leakage by splitting the data based on product IDs.
- Build TF-IDF and Logistic Regression baseline models.
- Fine-tune a pretrained Persian or multilingual Transformer.
- Compare classical machine learning and deep learning approaches.
- Apply pseudo-labeling to unlabeled training data.
- Evaluate models using appropriate classification metrics.
- Save trained models and preprocessing artifacts for inference.
- Provide an API for model predictions using FastAPI.

## Dataset

The project uses the public Digikala reviews dataset, containing approximately 7 million records.

**Dataset:** [Digikala Comments and Products](https://huggingface.co/datasets/RadeAI/Digikala_comments_products/tree/main)

The main model input is the review body (`body`).

The following targets are used:

| Task | Target | Classes |
| --- | --- | --- |
| Recommendation Classification | `recommendation_status` | `recommended`, `not_recommended`, `no_idea` |
| Rating-Band Classification | `rating_band` | `low`, `medium`, `high` |

The `rating_band` target is derived from the numeric `rate` column:

- `low`: 0 ≤ rate < 2.5
- `medium`: 2.5 ≤ rate < 4.0
- `high`: 4.0 ≤ rate ≤ 5.0

The raw rating and other metadata are not used as model input features.

## Project Workflow

1. **Data Understanding:** Dataset auditing, missing-value analysis, class distributions, and text-length analysis.
2. **Text Preprocessing:** Persian text normalization and cleaning.
3. **Data Splitting:** Deterministic product-grouped train, validation, and test splits.
4. **Baseline Models:** TF-IDF features with Logistic Regression.
5. **Transfer Learning:** Fine-tuning a pretrained Transformer for both classification tasks.
6. **Semi-Supervised Learning:** Generating pseudo-labels for selected unlabeled training reviews using a confidence threshold of 0.90.
7. **Evaluation:** Comparing models using Accuracy, Macro F1, Balanced Accuracy, per-class metrics, and confusion matrices.
8. **Inference:** Using saved artifacts to predict the labels of new reviews.
9. **API (Optional Extension):** Serving predictions through a FastAPI application.

## Repository Structure

```text
digikala_reviews_project/
│
├── app/
│   ├── main.py
│   └── model_loader.py
│   └── model.py
│   └── preproccessing.py
│
├── data
|    │
|    ├── token_cache/
|    │
|    ├── digikala-comments.csv
|    ├── digikala-comments-cleaned.csv
|    ├── digikala-comments-normalized.csv
|    ├── digikala-comments-final.csv
|    │
|    ├── digikala-products.csv
|    ├── digikala-products-cleaned.csv
|    │
|    ├── digikala-label-conflicts.csv
|    ├── digikala-label-conflicts-discarded.csv
|    ├── digikala-label-conflicts-resolved.csv
|    │
|    ├── train_batch_01.csv
|    ├── train_batch_02.csv
|    ├── train_batch_03.csv
|    ├── train_batch_04.csv
|    ├── train_batch_05.csv
|    ├── train_batch_06.csv
|    ├── train_batch_07.csv
|    ├── train_batch_08.csv
|    ├── train_batch_09.csv
|    ├── train_batch_10.csv
|    │
|    ├── train_experiment_100k.csv
|    ├── train_experiment_200k.csv
|    ├── train_experiment_300k.csv
|    ├── train_experiment_400k.csv
|    ├── train_experiment_500k.csv
|    ├── train_experiment_1000k.csv
|    │
|    ├── train_natural_unlabeled.csv
|    ├── test.csv
|    ├── test_10k.csv
|    ├── test_10k_corrected.csv
|    │
|    ├── reserved_product_ids.csv
|    ├── reserved_product_ids_before_test.csv
|    │
|    ├── accepted_pseudo_labels.csv
|    ├── pseudo_label_audit.csv
|    ├── pseudo_label_threshold_sweep.csv
|    │
|    ├── supervised_training_history.csv
|    ├── semi_supervised_training_history.csv
|    ├── semi_supervised_comparison.csv
|    ├── semi_supervised_final_summary.csv
|    └── semi_supervised_significance.csv
│
│
├── models/
│   ├── parsbert_multitask_best.pt
│   ├── parsbert_semi_supervised.pt
│   ├── parsbert_supervised_80.pt
│
│
├── notebooks/
│   ├── 01_descriptive_analysis.ipynb
│   ├── 02_data_cleaning_normalization.ipynb
│   ├── 03_data_size_experiment.ipynb
│   ├── 04_transfer_learning_finetuning.ipynb
│   └── 05_semi_supervised.ipynb
│
├── artifacts/ 
│    ├── semi_supervised/
│    ├── tfidf/
│    ├── transformer
│
├── report
│
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset and Data Directory

The project uses the Digikala customer reviews dataset, containing product information and customer review data.

The `data/` directory contains the original dataset, cleaned and normalized versions, training and testing splits, and experiment outputs.

### Data Categories

| Category | Description |
| --- | --- |
| Raw Data | Original Digikala comments and product datasets. |
| Cleaned Data | Reviews after data cleaning and preprocessing. |
| Normalized Data | Persian text after normalization. |
| Label Conflicts | Files used to identify, discard, and resolve conflicting labels. |
| Training Batches | Training data divided into smaller batches. |
| Training Experiments | Training subsets with different sample sizes, from 100K to 1M records. |
| Unlabeled Data | Naturally unlabeled reviews used in the semi-supervised experiment. |
| Test Data | Separate datasets used for model evaluation. |
| Reserved Product IDs | Product identifiers used to manage product-level data separation. |
| Pseudo-Labeling | Generated pseudo-labels, audit records, and threshold experiments. |
| Training History | Training logs and experiment comparison results. |
| Token Cache | Cached tokenization data used during model processing.

### Data Processing and Splitting

The data preparation workflow includes:

1. Loading the original Digikala comments and product datasets.
2. Cleaning and normalizing review text.
3. Identifying and resolving conflicting labels.
4. Creating training subsets for different experiment sizes.
5. Separating training and test data based on product IDs to reduce data leakage.
6. Preparing naturally unlabeled reviews for semi-supervised learning.
7. Saving pseudo-labeling results and experiment logs.

The test datasets are kept separate from the training process and are used for final model evaluation.

### Dataset Source

[Digikala Comments and Products Dataset](https://huggingface.co/datasets/RadeAI/Digikala_comments_products/tree/main)

Large CSV files and intermediate data files may be excluded from Git tracking when necessary. The repository documentation should specify how to obtain the required data files.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd digikala_reviews_project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** PyTorch installation may differ depending on whether you use a CPU or a supported CUDA GPU. Follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) if the default installation is not suitable for your machine.

## Dataset and Model Artifacts

Download the dataset from the link provided above and place the required files in the appropriate data directory.

Trained model checkpoints and other large artifacts may be stored using Git LFS or an external model repository. They are not necessarily included directly in the Git repository.

Make sure the required model and tokenizer files are available before running inference or starting the API.

## Running the FastAPI Service

The API is an optional deployment extension.

From the repository root, run:

```bash
uvicorn app.main:app --reload
```

Then open:

- Health check: `http://127.0.0.1:8000/health`
- API documentation: `http://127.0.0.1:8000/docs`

## Evaluation

The models are evaluated using metrics appropriate for imbalanced classification tasks:

- Accuracy
- Macro F1
- Balanced Accuracy
- Per-class Precision, Recall, and F1-score
- Confusion Matrix

For the ordered `rating_band` task, ordinal distance error is also considered.

The semi-supervised experiment compares the supervised and pseudo-label-trained models on the same validation and test sets.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- FastAPI
- Uvicorn
- Jupyter Notebook

## Project Status

This project is under development.

The modeling pipeline focuses on the required classification and semi-supervised learning tasks. The FastAPI service is an additional deployment component.
