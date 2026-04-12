"""Compatibility shim: re-export functions from inference.py expected
by app/inference_service which imports 'interface'.
"""
from inference import (
    load_models,
    load_data_and_prepare,
    run_dom_extractor,
    run_field_classifier,
    predict_events,
)

__all__ = [
    "load_models",
    "load_data_and_prepare",
    "run_dom_extractor",
    "run_field_classifier",
    "predict_events",
]
