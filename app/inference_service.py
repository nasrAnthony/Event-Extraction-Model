import io
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - environment-specific optional dependency
    torch = None

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models"
# Candidate artifact locations: prefer models/ but also accept repo-root files
CHECKPOINT_CANDIDATES = [MODEL_DIR / "dom_extractor_checkpoint.pt", ROOT / "dom_extractor_checkpoint_v2.pt", MODEL_DIR / "dom_extractor_checkpoint_v2.pt"]
CLASSIFIER_CANDIDATES = [MODEL_DIR / "field_classifier_v1.joblib", ROOT / "field_classifier_v1.joblib"]


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def _get_checkpoint_path() -> Path:
    return _first_existing(CHECKPOINT_CANDIDATES)


def _get_classifier_path() -> Path:
    return _first_existing(CLASSIFIER_CANDIDATES)


def _load_pipeline_functions():
    from interface import (
        load_models,
        load_data_and_prepare,
        run_dom_extractor,
        run_field_classifier,
        predict_events,
    )

    return {
        "load_models": load_models,
        "load_data_and_prepare": load_data_and_prepare,
        "run_dom_extractor": run_dom_extractor,
        "run_field_classifier": run_field_classifier,
        "predict_events": predict_events,
    }


def get_device() -> str:
    if torch is None:
        return "cpu (torch unavailable)"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_artifact_status() -> dict:
    ck = _get_checkpoint_path()
    cf = _get_classifier_path()
    return {
        "checkpoint_exists": ck.exists(),
        "classifier_exists": cf.exists(),
        "checkpoint_path": str(ck),
        "classifier_path": str(cf),
        "device": get_device(),
        "torch_available": torch is not None,
        "real_model_ready": ck.exists() and cf.exists() and torch is not None,
    }


def load_model() -> Optional[dict]:
    """
    Load the full model bundle required by the real inference pipeline.
    Returns None if artifacts are missing or loading fails.
    """
    ckpt_path = _get_checkpoint_path()
    classifier_path = _get_classifier_path()

    if torch is None or not ckpt_path.exists() or not classifier_path.exists():
        return None

    device = get_device()

    try:
        pipeline = _load_pipeline_functions()
        model, tokenizer, ckpt, field_bundle = pipeline["load_models"](
            checkpoint_path=str(ckpt_path),
            classifier_path=str(classifier_path),
            device=device,
        )
        return {
            "model": model,
            "tokenizer": tokenizer,
            "ckpt": ckpt,
            "field_bundle": field_bundle,
            "device": device,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {e}") from e


def get_required_columns(model_bundle: Optional[dict] = None) -> list[str]:
    base_required = ["rendering_order", "text_context", "tag", "parent_tag"]

    if model_bundle is None:
        return base_required

    ckpt = model_bundle["ckpt"]
    extra_required = ckpt.get("num_cols", []) + ckpt.get("bool_cols", [])
    return base_required + extra_required


def validate_input(df: pd.DataFrame, model_bundle: Optional[dict] = None) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, "Uploaded file is empty."

    required = get_required_columns(model_bundle)
    missing = [c for c in required if c not in df.columns]

    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    return True, ""


def run_inference(df: pd.DataFrame, model: Optional[object] = None) -> pd.DataFrame:
    """Run a lightweight inference pass using a sklearn-style model/pipeline.

    Returns the input DataFrame with an added `is_event_pred` column.
    This is useful for tests and small demo pipelines (and complements the
    heavier `extract_events_from_df` pipeline used for the full models).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if model is None:
        out["is_event_pred"] = 0
        return out

    preds = None
    try:
        preds = model.predict(out)
    except Exception:
        try:
            if "text_context" in out.columns:
                X = out["text_context"].fillna("").astype(str)
                preds = model.predict(X)
            else:
                preds = model.predict(out.iloc[:, :1])
        except Exception:
            preds = None

    if preds is None:
        out["is_event_pred"] = 0
        return out

    # handle probability arrays, numeric labels or string labels
    try:
        import numpy as np

        arr = np.asarray(preds)
        if arr.ndim == 2 and arr.shape[1] > 1:
            # probabilities per class; assume class 1 is positive
            probs = arr[:, 1].astype(float)
            out["is_event_pred"] = (probs >= 0.5).astype(int)
            return out
    except Exception:
        pass

    ser = pd.Series(preds)
    if ser.dtype == object:
        low = ser.fillna("").astype(str).str.lower()
        positive = low.isin(["event", "events", "1", "true", "yes", "y", "positive", "pos"])
        numeric = pd.to_numeric(low, errors="coerce").fillna(0).astype(int)
        out["is_event_pred"] = (positive | (numeric == 1)).astype(int)
    else:
        out["is_event_pred"] = pd.to_numeric(ser, errors="coerce").fillna(0).astype(int)

    return out


def extract_events_from_df(df: pd.DataFrame, model_bundle: Optional[dict] = None) -> list[dict]:
    """
    Run the real event extraction pipeline on an uploaded dataframe.
    """
    if df is None or df.empty:
        return []

    if model_bundle is None:
        raise ValueError("Model bundle is not loaded.")

    temp_path = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        temp_path = Path(tmp.name)
        df.to_csv(temp_path, index=False)

    try:
        pipeline = _load_pipeline_functions()

        page_df = pipeline["load_data_and_prepare"](
            csv_path=str(temp_path),
            ckpt=model_bundle["ckpt"],
            source_name="uploaded_file",
        )

        dom_results = pipeline["run_dom_extractor"](
            page_df=page_df,
            model=model_bundle["model"],
            tokenizer=model_bundle["tokenizer"],
            ckpt=model_bundle["ckpt"],
            device=model_bundle["device"],
        )

        node_labels = pipeline["run_field_classifier"](
            page_df=page_df,
            dom_results=dom_results,
            field_bundle=model_bundle["field_bundle"],
        )

        events = pipeline["predict_events"](
            page_df=page_df,
            dom_results=dom_results,
            node_labels=node_labels,
        )

        return events

    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def format_events_for_ui(events: list[dict]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    if "event_index" not in df.columns and "source" in df.columns:
        df["event_index"] = df.groupby("source").cumcount()

    preferred = ["source", "event_number", "event_index"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def dataframe_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
