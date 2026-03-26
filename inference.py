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


def load_models(checkpoint_path, classifier_path, device):
    """Load both models"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]["model"]

    # Boundary Model
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

    # classifier
    field_bundle = joblib.load(classifier_path)

    return model, tokenizer, ckpt, field_bundle


def load_data_and_prepare(csv_path, ckpt, source_name=None):
    """
    deal with missing columns
    inputs:
        - csv file path
        - checkpoint from loaded model
        - source name can be used for single page csv files
    output: dataframe ready for models
    """
    df = pd.read_csv(csv_path)
    
    # handle source column
    if "source" not in df.columns:
        name = source_name or Path(csv_path).stem
        df["source"] = name
    
    # verify required columns
    required = ["rendering_order", "text_context", "tag", "parent_tag"] + \
               ckpt["num_cols"] + ckpt["bool_cols"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"See README for full list of required columns."
        )
    
    # sort within each source (extra safety)
    df = df.sort_values(["source", "rendering_order"]).reset_index(drop=True)
    
    return df


@torch.no_grad()
def run_dom_extractor(page_df, model, tokenizer, ckpt, device):
    """
    function to run the model
    inputs:  
    """
    cfg = ckpt["cfg"]
    inf = cfg["inference"]

    # keep sources in same order
    sources = list(page_df.groupby("source", sort=False).groups.keys())

    # make dataset and loader for the dom model
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

    results = []
    for i, batch in enumerate(loader):
        enc = {k: v.to(device) for k, v in batch["enc"].items()}
        node_mask = batch["node_mask"].to(device).bool()

        # forward pass
        bio_logits = model(
            enc=enc,
            node_offsets=batch["node_offsets"],
            node_mask=node_mask,
            tag_id=batch["tag_id"].to(device),
            parent_tag_id=batch["parent_tag_id"].to(device),
            num_feats=batch["num_feats"].to(device),
            bool_feats=batch["bool_feats"].to(device),
        )

        # get indices of real nodes
        valid = torch.where(node_mask[0])[0].cpu().numpy()

        # convert logits to probabilities
        probs_full = torch.softmax(bio_logits, dim=-1)[0].cpu().numpy()
        prob_B = probs_full[:, 1]
        prob_B_valid = prob_B[valid]
        
        # find predicted event starts using prob_B and threshold
        probs_full_valid = probs_full[valid]

        start_indices = pick_starts_from_probs(
            prob_B_valid,
            threshold=ckpt["best_th"],
            nms_k=inf["nms_k"],
            min_gap=inf["min_gap"],
        )

        for s in start_indices:
            probs_full_valid[s] = [0, 1, 0]  # force B

        if len(start_indices) == 0:
            print(f"Warning: no events found for page {i}")
            results.append([])
            continue
        
        results.append({
            "source": sources[i],
            "probs_full_valid": probs_full_valid,
            "valid": valid,
            "start_indices": start_indices
        })

    return results


def run_field_classifier(page_df, dom_results, field_bundle):
    """
    Runs field classifier on all B+I nodes for each page.
    Returns list of dicts mapping page-local node index -> predicted field label.
    """
    all_node_labels = []
    
    for result in dom_results:
        if result is None:
            all_node_labels.append(None)
            continue
        
        source = result["source"]
        probs_full_valid = result["probs_full_valid"]
        
        # get page slice
        page = page_df[page_df["source"] == source].sort_values("rendering_order").reset_index(drop=True)
        
        # get indices of B+I nodes (not O)
        labels_bio = probs_full_valid.argmax(axis=1)  # 0=O, 1=B, 2=I
        bi_indices = [i for i, l in enumerate(labels_bio) if l != 0]
        
        if len(bi_indices) == 0:
            all_node_labels.append({})
            continue
        
        # run classifier on all B+I nodes at once
        bi_df = page.iloc[bi_indices]
        X = build_features(
            df=bi_df,
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
        field_labels = field_bundle["label_encoder"].inverse_transform(preds)
        
        # map page-local index -> field label
        node_labels = {bi_indices[j]: field_labels[j] for j in range(len(bi_indices))}
        all_node_labels.append(node_labels)
    
    return all_node_labels


def predict_events(page_df, dom_results, node_labels):
    """
    Takes DOM extractor results and node field labels,
    corrects predicted starts and builds event spans.
    Returns list of formatted event dicts.
    """
    all_events = []
    
    for result, labels in zip(dom_results, node_labels):
        if not result or labels is None:
            continue
        
        source = result["source"]
        probs_full_valid = result["probs_full_valid"]
        valid = result["valid"]
        start_indices = list(result["start_indices"])  # in valid-node space
        bio_labels = probs_full_valid.argmax(axis=1)   # 0=O, 1=B, 2=I
        N = len(bio_labels)
        
        # get page slice for text lookup later
        page = page_df[page_df["source"] == source].sort_values("rendering_order").reset_index(drop=True)

        # step 1: first start correction — check node above in valid-node space
        first = start_indices[0]
        if first > 0 and bio_labels[first - 1] != 0:  # node above is I or B
            start_indices[0] = first - 1
            bio_labels[first] = 2  # change old start from B to I

        # step 2: get reference field from first start
        # labels keys are in valid-node space
        reference_field = labels.get(start_indices[0])
        
        if reference_field is None or reference_field == "Other":
            print(f"Warning: could not determine reference field for {source}")
            continue

        # step 3: validate remaining starts
        validated_starts = [start_indices[0]]
        
        for start in start_indices[1:]:
            field = labels.get(start)
            
            if field == reference_field:
                validated_starts.append(start)
            elif labels.get(start - 1) == reference_field:
                validated_starts.append(start - 1)
                bio_labels[first] = 2  # change old start from B to I
            elif labels.get(start + 1) == reference_field:
                validated_starts.append(start + 1)
                bio_labels[first] = 2  # change old start from B to I
            else:
                print(f"Warning: discarding invalid start at node {start} for {source}")

        # step 4: build spans in valid-node space
        spans = []
        for i, start in enumerate(validated_starts):
            if i + 1 < len(validated_starts):
                end = validated_starts[i + 1]
            else:
                # last event — find last I node
                end = start
                for j in range(start, N):
                    if bio_labels[j] == 2:
                        end = j
                end = end + 1
            spans.append((start, end))

        # step 5: format events — map valid-node indices to page-local via valid[]
        for event_num, (start, end) in enumerate(spans, start=1):
            event = {"source": source, "event_number": event_num}
            field_counts = {}
            
            for valid_idx in range(start, end):
                label = labels.get(valid_idx)
                if label is None or label == "Other":
                    continue
                page_idx = int(valid[valid_idx])  # map to page-local index
                field_counts[label] = field_counts.get(label, 0) + 1
                key = f"{label}_{field_counts[label]}"
                event[key] = str(page.iloc[page_idx]["text_context"])
            
            all_events.append(event)
    
    return all_events


def save_output(events, output_path):
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)
