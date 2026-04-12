# Event Extraction Model

Event Extraction Model converts webpage DOM node data into structured event records. The project combines a DOM-aware boundary detector with a field classifier to extract event attributes such as `Name`, `Date`, `Time`, `Location`, `Description`, and `Price` from page content that has already been exported to CSV.

## Overview

The pipeline operates in two stages:

1. A DOM-aware sequence model reads nodes in rendering order and predicts BIO tags (`O`, `B`, `I`) to find event boundaries.
2. A field classifier labels nodes inside those event spans with semantic event fields.

The final output is a flat event record with repeat-safe keys such as `Date_1`, `Location_2`, and `Time_1`.

## Repository Layout

```text
Event-Extraction-Model/
├── app/                    # Streamlit app and inference-service wrapper
├── config.yaml             # Model, training, inference, and feature configuration
├── data/                   # Raw, cleaned, and compiled datasets
├── helpers/                # Cleaning, concat, dataset, metrics, and training helpers
├── models/                 # DOM extractor, field classifier, and experiments
├── tests/                  # Production-facing tests and manual UI fixtures
├── inference.py            # Reusable inference pipeline
├── train_dom_extractor.py  # Boundary model training script
└── requirements.txt        # Pinned Python dependencies
```

## Installation

Create a virtual environment, activate it, and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The pretrained inference path requires `torch`, `transformers`, `joblib`, and `scikit-learn` in the same environment as Streamlit. The first real-model run may also download the Hugging Face assets for `distilbert-base-uncased` if they are not already cached.

## Dataset

The repository already contains a compiled training dataset in `data/full_data.csv`.

- Total DOM nodes: `3100`
- Labeled sources: `15`
- Labeled events: `165`

The source-by-source data is stored in `data/raw/` and `data/cleaned/`.

## Input Schema

### Core Inference Columns

All inference inputs need these base columns:

- `rendering_order`
- `text_context`
- `tag`
- `parent_tag`

When using the real pretrained model, the input also needs the engineered numeric and boolean feature columns stored in the checkpoint metadata. The easiest way to test that path is to use the sample files in `tests/fixtures/streamlit/` or a page exported in the same schema as `data/full_data.csv`.

## Training

### Train The DOM Boundary Model

```bash
python3 train_dom_extractor.py
```

This trains the boundary model, tunes the boundary threshold, evaluates on a holdout set, and saves the DOM checkpoint artifact.

### Train The Field Classifier

```bash
python3 models/field_classifier.py
```

This trains the field classifier on labeled event nodes and saves the classifier bundle.

## Inference

Reusable inference functions live in `inference.py`.

High-level flow:

1. Load the DOM checkpoint and field-classifier bundle.
2. Validate and sort the input rows by `source` and `rendering_order`.
3. Predict BIO boundaries for each page.
4. Run the field classifier on predicted event nodes.
5. Assemble flat event dictionaries.

The Streamlit UI for local inference lives in `app/streamlit_app.py`.

## Artifacts

The local app workflow looks for these pretrained artifact filenames:

- `dom_extractor_checkpoint_v2.pt`
- `field_classifier_v1.joblib`

At the moment, `field_classifier_v1.joblib` is tracked in Git. The DOM checkpoint may exist locally or under `models/`, depending on how the environment was prepared.

## Tests

Production-facing tests live in `tests/`.

Run the suite with:

```bash
python3 -m pytest -q
```

Branch-specific UI/testing workflow notes live in `docs/ui-testing-integration.md`.
