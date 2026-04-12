# UI Testing Integration Notes

This document captures the branch-specific workflow for `ui-testing-integration`.

## Run Tests

```bash
python3 -m pytest -q
```

## Run The Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

## Streamlit Upload Samples

Use the files in `tests/fixtures/streamlit/`:

- `sample_for_ui_min.csv`
  - smallest demo-mode smoke test
  - expected extracted events: `1`
- `sample_for_ui_pretrained_hawaii_window.csv`
  - compact pretrained-model smoke test
  - expected extracted events: `3`
- `sample_for_ui_pretrained_hawaii_page.csv`
  - fuller pretrained-model smoke test
  - expected extracted events: `7`

## Pretrained Model Notes

To test the real pretrained inference path locally, make sure these packages are installed in the same environment as Streamlit:

- `torch`
- `transformers`
- `joblib`
- `scikit-learn`

The app looks for these artifact filenames locally:

- `dom_extractor_checkpoint_v2.pt`
- `field_classifier_v1.joblib`

## Emulate CI Locally

The GitHub Actions workflow runs a lightweight test job. A close local equivalent is:

```bash
python3 -m venv .venv-ci
source .venv-ci/bin/activate
python -m pip install --upgrade pip
pip install pandas pytest streamlit scikit-learn joblib
export PYTHONPATH="$(pwd)"
python -m pytest -q
```

## Current Verification Notes

On this branch, the following checks have already been verified locally:

- `python3 -m pytest -q` passes
- the Streamlit app boots successfully
- the demo sample returns `1` event
- the pretrained Hawaii window sample returns `3` events
- the pretrained Hawaii page sample returns `7` events
- the lightweight CI workflow passes in a clean temporary virtual environment
