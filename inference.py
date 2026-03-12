import json
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from helpers.dataset import PageDataset, combine_pages
from helpers.metrics import pick_starts_from_probs
from models.dom_extractor import DOMAwareEventExtractor
from models.field_classifier import build_features

ROOT = Path(__file__).resolve().parent


def load_models(checkpoint_path, classifier_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]["model"]

    model = DOMAwareEventExtractor(
        text_model_name=cfg["name"],
        tag_vocab_size=len(ckpt["tag_vocab"]),
        parent_tag_vocab_size=len(ckpt["parent_tag_vocab"]),
        num_numeric_features=len(ckpt["num_cols"]),
        num_bool_features=len(ckpt["bool_cols"]),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_tag=cfg.get("use_tag", True),
        use_parent_tag=cfg.get("use_parent_tag", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    field_bundle = joblib.load(classifier_path)

    return model, tokenizer, ckpt, field_bundle

def get_page(df, source):
    page_df = df[df["source"] == source].sort_values("rendering_order").reset_index(drop=True)
    if page_df.empty:
        raise ValueError(f"Source '{source}' not found in dataframe")
    return page_df


@torch.no_grad()
def predict_events(page_df, model, tokenizer, ckpt, field_bundle, device):
    cfg = ckpt["cfg"]
    inf = cfg["inference"]

    dataset = PageDataset(
        df=page_df,
        tokenizer=tokenizer,
        tag_vocab=ckpt["tag_vocab"],
        parent_tag_vocab=ckpt["parent_tag_vocab"],
        num_cols=ckpt["num_cols"],
        bool_cols=ckpt["bool_cols"],
        mean=ckpt["num_mean"],
        std=ckpt["num_std"],
        max_tokens=cfg["model"]["max_tokens"],
    )

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=False,
        collate_fn=partial(combine_pages, tokenizer=tokenizer)
    )

    for batch in loader:
        enc = {k: v.to(device) for k, v in batch["enc"].items()}
        node_mask = batch["node_mask"].to(device).bool()
        bio_logits = model(
            enc=enc,
            node_offsets=batch["node_offsets"],
            node_mask=node_mask,
            tag_id=batch["tag_id"].to(device),
            parent_tag_id=batch["parent_tag_id"].to(device),
            num_feats=batch["num_feats"].to(device),
            bool_feats=batch["bool_feats"].to(device),
        )
        valid = torch.where(node_mask[0])[0].cpu().numpy()
        prob_B = torch.softmax(bio_logits, dim=-1)[0, :, 1].cpu().numpy()
        prob_B_valid = prob_B[valid]

    start_indices = pick_starts_from_probs(
        prob_B_valid, threshold=ckpt["best_th"],
        nms_k=inf["nms_k"], min_gap=inf["min_gap"]
    )
    start_indices = [int(valid[i]) for i in start_indices]

    events = []
    for i, start in enumerate(start_indices):
        end = start_indices[i + 1] if i + 1 < len(start_indices) else len(page_df)
        event_df = page_df.iloc[start:end]

        X = build_features(
            df=event_df,
            tfidf=field_bundle["tfidf"],
            tag_columns=field_bundle["tag_columns"],
            parent_columns=field_bundle["parent_columns"],
            num_cols=field_bundle["num_cols"],
            bool_cols=field_bundle["bool_cols"],
            fit=False,
            use_tag=field_bundle.get("use_tag", True),
            use_parent_tag=field_bundle.get("use_parent_tag", True),
        )
        preds = field_bundle["clf"].predict(X)
        labels = field_bundle["label_encoder"].inverse_transform(preds)

        fields = {}
        for label, text in zip(labels, event_df["text_context"].astype(str)):
            if label not in fields:
                fields[label] = text

        events.append({
            "event_index": i,
            "node_range": [start, end],
            "fields": fields,
        })

    return events


def save_output(events, output_path):
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)
