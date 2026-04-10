# Event Extraction Model

Event Extraction Model converts webpage DOM node data into structured event records. The project combines a DOM-aware boundary detector with a field classifier to extract event-level attributes such as `Name`, `Date`, `Time`, `Location`, `Description`, and `Price` from page content that has already been exported to CSV.

## Overview

The pipeline operates in two stages:

1. A DOM-aware sequence model reads nodes in rendering order and predicts BIO tags (`O`, `B`, `I`) to find event boundaries.
2. A field classifier labels the nodes inside those event spans with semantic event fields.

The final output is a JSON array of flat event records with repeat-safe keys such as `Date_1`, `Location_2`, and `Time_1`.

## What The Project Delivers

- Extracts event records from DOM-derived webpage node data.
- Learns event boundaries at page level instead of classifying nodes independently.
- Uses text, DOM structure, HTML tag context, and engineered page features together.
- Normalizes label variants into a fixed field taxonomy.
- Saves reusable model artifacts for repeatable inference.

## Dataset Snapshot

The repository already contains a compiled training dataset in [data/full_data.csv](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/data/full_data.csv).

- Total DOM nodes: `3100`
- Labeled sources: `15`
- Labeled events: `165`
- Field-labeled nodes after normalization: `662`
- Normalized field classes: `Date`, `Description`, `Location`, `Name`, `Price`, `Time`

Field-label distribution after normalization:

| Label | Count |
| --- | ---: |
| Date | 242 |
| Location | 139 |
| Name | 120 |
| Time | 116 |
| Description | 27 |
| Price | 18 |

## Architecture

### 1. DOM Boundary Extractor

The boundary extractor in [models/dom_extractor.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/models/dom_extractor.py) is the primary model in the pipeline.

- Text encoder: `distilbert-base-uncased`
- Page encoder: Transformer encoder over DOM node sequence
- Prediction target: BIO tags for each node
- Features fused into each node representation:
  - DistilBERT `[CLS]` embedding for `text_context`
  - HTML `tag` embedding
  - `parent_tag` embedding
  - Numeric DOM features
  - Boolean DOM/text features

Training and evaluation logic lives in [train_dom_extractor.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/train_dom_extractor.py).

### 2. Field Classifier

The field classifier in [models/field_classifier.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/models/field_classifier.py) runs on nodes predicted as part of an event.

- Model: `GradientBoostingClassifier`
- Text features: TF-IDF with up to 300 features and 1-2 gram vocabulary
- Structural features:
  - Numeric DOM features from `config.yaml`
  - Boolean DOM/text features from `config.yaml`
  - One-hot encoded `tag`
  - One-hot encoded `parent_tag`

### 3. Event Assembly

The inference pipeline in [inference.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/inference.py) performs the final assembly:

- Loads the DOM checkpoint and field-classifier bundle.
- Validates and sorts input rows by `source` and `rendering_order`.
- Predicts BIO probabilities for each page.
- Selects event starts using thresholded peak picking.
- Labels event nodes with field names.
- Builds final event spans and serializes flat JSON records.

## Repository Layout

```text
Event-Extraction-Model/
├── config.yaml                  # Model, training, inference, and feature configuration
├── train_dom_extractor.py       # Boundary model training script
├── inference.py                 # Reusable inference functions
├── requirements.txt             # Pinned Python dependencies
├── data/
│   ├── raw/                     # Original labeled source files
│   ├── cleaned/                 # Cleaned versions of the raw source files
│   └── full_data.csv            # Combined training dataset
├── helpers/
│   ├── clean_data.py            # CSV cleaning utility
│   ├── concat.py                # Dataset concatenation utility
│   ├── dataset.py               # Label normalization, page dataset, collation
│   ├── losses.py                # BIO loss functions
│   ├── metrics.py               # Boundary decoding and evaluation metrics
│   ├── train_utils.py           # DataLoader and epoch helpers
│   └── utils.py                 # Config loading and numeric feature statistics
├── models/
│   ├── dom_extractor.py         # DOM-aware event extractor model
│   ├── field_classifier.py      # Field classifier training script
│   ├── classifier_model.py      # Separate CatBoost baseline experiment
│   └── catboost_info/           # CatBoost training artifacts
├── field_classifier_v1.joblib   # Exported field-classifier bundle checked into the repo
├── predicted_events.json        # Sample serialized prediction output
└── *.ipynb                      # Notebook-based experimentation and analysis
```

## Installation

Create a virtual environment, activate it, and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The DOM boundary model uses PyTorch and runs on CUDA when available, otherwise on CPU. The code selects the device automatically.

The first run of the boundary model downloads the Hugging Face tokenizer and model weights for `distilbert-base-uncased` unless they are already cached.

## Configuration

All core settings live in [config.yaml](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/config.yaml).

Current defaults:

- Backbone model: `distilbert-base-uncased`
- DOM transformer width: `128`
- Attention heads: `4`
- Transformer layers: `2`
- Dropout: `0.2`
- Max tokens per DOM node: `64`
- Training epochs: `20`
- Frozen BERT warm-up epochs: `4`
- Batch size: `2`
- BERT learning rate: `5e-6`
- Non-BERT learning rate: `1e-4`
- Weight decay: `0.01`
- Cross-validation folds: `5`
- Threshold decoding:
  - `nms_k = 1`
  - `min_gap = 2`
  - `tol = 1`

## Data Pipeline

### Raw And Cleaned Source Files

The dataset is stored source-by-source in both [data/raw](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/data/raw) and [data/cleaned](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/data/cleaned).

Cleaning is handled by [helpers/clean_data.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/helpers/clean_data.py):

- Removes exact duplicate rows
- Strips whitespace from key string columns
- Casts `link` to string dtype

Run it with:

```bash
python3 helpers/clean_data.py
```

### Combined Training Dataset

[helpers/concat.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/helpers/concat.py) concatenates all cleaned source files into a single training file and injects a `source` column from the filename stem.

Run it with:

```bash
python3 helpers/concat.py
```

## Label Taxonomy

The project normalizes label variants in [helpers/dataset.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/helpers/dataset.py) before model training.

Merged label mapping:

| Original Labels | Normalized Label |
| --- | --- |
| `Name`, `NameLink`, `NameLocation` | `Name` |
| `Date`, `DateTime` | `Date` |
| `Time`, `StartTime`, `EndTime`, `StartEndTime`, `TimeLocation` | `Time` |
| `Location` | `Location` |
| `Price` | `Price` |
| `Description`, `Desc`, `Details` | `Description` |
| Any unmapped label | `Other` |

## Input Schema

### Training Schema

The full training file includes these columns:

```text
rendering_order
tag
attributes
text_context
depth
parent_index
parent_tag
text_length
sibling_index
link
children_count
same_tag_sibling_count
same_text_sibling_count
has_link
link_is_absolute
parent_has_link
is_leaf
word_count
letter_ratio
digit_ratio
whitespace_ratio
contains_date
contains_time
starts_with_digit
ends_with_digit
attribute_count
has_class
has_id
attr_has_word_name
attr_has_word_date
attr_has_word_time
attr_has_word_location
attr_has_word_link
text_has_word_name
text_has_word_date
text_word_time
text_word_description
text_word_location
text_word_am
text_word_pm
label
event_id
source
```

### Inference Schema

`load_data_and_prepare()` in [inference.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/inference.py) validates the inference input against the checkpoint metadata and the current config-driven feature set.

Required core columns:

- `rendering_order`
- `text_context`
- `tag`
- `parent_tag`

Required numeric features:

- `depth`
- `sibling_index`
- `children_count`
- `same_tag_sibling_count`
- `same_text_sibling_count`
- `text_length`
- `word_count`
- `letter_ratio`
- `digit_ratio`
- `whitespace_ratio`
- `attribute_count`

Required boolean features:

- `has_link`
- `link_is_absolute`
- `parent_has_link`
- `is_leaf`
- `contains_date`
- `contains_time`
- `starts_with_digit`
- `ends_with_digit`
- `has_class`
- `has_id`
- `attr_has_word_name`
- `attr_has_word_date`
- `attr_has_word_time`
- `attr_has_word_location`
- `attr_has_word_link`
- `text_has_word_name`
- `text_has_word_date`
- `text_word_time`
- `text_word_description`
- `text_word_location`

Additional schema rules:

- `source` is optional during inference. If it is missing, the loader derives it from the CSV filename or from the explicit `source_name` argument.
- Rows are sorted by `source` and `rendering_order` before inference.
- The current pipeline does not consume `attributes`, `link`, `parent_index`, `text_word_am`, or `text_word_pm`.

## Training

### Train The DOM Boundary Model

Run:

```bash
python3 train_dom_extractor.py
```

This script:

- Loads `data/full_data.csv`
- Normalizes field labels
- Builds binary event membership and BIO start labels
- Splits data by `source` using `GroupShuffleSplit`
- Performs source-level cross-validation on the training partition
- Sweeps boundary thresholds to maximize F1
- Retrains on the full training partition
- Evaluates on a holdout test partition
- Saves `models/dom_extractor_checkpoint.pt`

The checkpoint includes:

- Model weights
- Label vocabulary
- HTML tag vocabularies
- Numeric feature means and standard deviations
- Feature-column lists
- Best threshold from cross-validation
- Full config used for the run

### Train The Field Classifier

Run:

```bash
python3 models/field_classifier.py
```

This script:

- Loads `data/full_data.csv`
- Applies label normalization
- Filters out `Other`
- Encodes the six target field classes
- Builds TF-IDF, numeric, boolean, tag, and parent-tag features
- Evaluates with a train/test split and optional K-fold cross-validation
- Saves `models/field_classifier.joblib`

The saved bundle includes:

- Fitted `GradientBoostingClassifier`
- Fitted `TfidfVectorizer`
- `LabelEncoder`
- Tag and parent-tag column definitions
- Numeric and boolean feature-column lists
- Feature toggles for `use_tag` and `use_parent_tag`

### Checked-In Artifacts

The repository currently contains:

- [field_classifier_v1.joblib](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/field_classifier_v1.joblib), an exported field-classifier bundle
- [predicted_events.json](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/predicted_events.json), a sample prediction export

Fresh training writes new artifacts into `models/`.

## Inference

Inference is exposed as reusable Python functions in [inference.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/inference.py). The committed code does not define a standalone CLI entry point.

### End-To-End Example

```python
import torch

from inference import (
    load_models,
    load_data_and_prepare,
    run_dom_extractor,
    run_field_classifier,
    predict_events,
    save_output,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer, ckpt, field_bundle = load_models(
    checkpoint_path="models/dom_extractor_checkpoint.pt",
    classifier_path="models/field_classifier.joblib",
    device=device,
)

page_df = load_data_and_prepare(
    csv_path="path/to/page_nodes.csv",
    ckpt=ckpt,
    source_name="my_page",
)

dom_results = run_dom_extractor(page_df, model, tokenizer, ckpt, device)
node_labels = run_field_classifier(page_df, dom_results, field_bundle)
events = predict_events(page_df, dom_results, node_labels)

save_output(events, "predicted_events.json")
```

### Inference Output Contract

`predict_events()` returns a list of dictionaries. Each dictionary contains:

- `source`
- `event_number`
- One or more extracted field entries named as `<Field>_<Index>`

Example shape:

```json
[
  {
    "source": "pnacac_spring.org_pattern_labeled",
    "event_number": 1,
    "Name_1": "Spring College Fair",
    "Date_1": "April 16, 2026",
    "Location_1": "Student Center",
    "Time_1": "6:00 PM - 8:00 PM"
  }
]
```

Repeated fields stay separate by index. This preserves every extracted node instead of collapsing duplicate field labels into a single value.

## Evaluation

### Boundary Evaluation

Boundary evaluation logic is implemented in [helpers/metrics.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/helpers/metrics.py).

The boundary model is scored with:

- Precision
- Recall
- F1

Decoding details:

- The model predicts BIO probabilities for every valid node on the page.
- Candidate event starts are selected from the `B` probability stream.
- Threshold search sweeps values from `0.01` to `0.199` in `0.001` increments.
- Peak selection keeps the leftmost local maximum inside each above-threshold region.
- `min_gap` enforces spacing between consecutive event starts.
- `tol=1` allows a predicted start to match a true start within one node.

### Field Classifier Evaluation

The field classifier reports:

- Full classification report
- Micro F1
- Macro F1
- Confusion matrix

## Additional Baseline

[models/classifier_model.py](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/models/classifier_model.py) contains a separate CatBoost-based binary event classifier experiment. It is not part of the main two-stage extraction pipeline.

That script:

- Predicts `is_event` at node level
- Uses `tag` as categorical input
- Uses `text_context` as text input
- Splits pages by `source`
- Saves a CatBoost model to `classifier.cbm`

`catboost` is required for this script and is not included in the current `requirements.txt`.

## Notebooks

The repository includes notebooks that mirror the scripted workflows and support model exploration:

- [data.ipynb](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/data.ipynb)
- [train-dom-extractor.ipynb](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/train-dom-extractor.ipynb)
- [field-classifier.ipynb](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/field-classifier.ipynb)
- [inference.ipynb](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/inference.ipynb)
- [test-playground.ipynb](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/test-playground.ipynb)
- [models/train.ipynb](/Users/christine/Documents/PythonProjects/Event-Extraction-Model/Event-Extraction-Model/models/train.ipynb)

## Summary

This repository defines a complete event-extraction workflow for DOM-derived webpage data:

- Data cleaning and dataset assembly
- DOM-aware boundary detection
- Field-level node classification
- Structured JSON event generation
- Reproducible training configuration and saved artifacts
