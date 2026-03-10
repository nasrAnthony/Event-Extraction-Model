import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from helpers.train_utils import make_loader
from helpers.metrics import collect_page_probs_and_truth, pick_starts_from_probs
from models.dom_extractor import DOMAwareEventExtractor


def load_checkpoint(checkpoint_path):
    """load checkpoint and rebuild the model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["cfg"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    model = DOMAwareEventExtractor(
        text_model_name=cfg["model"]["name"],
        tag_vocab_size=len(checkpoint["tag_vocab"]),
        parent_tag_vocab_size=len(checkpoint["parent_tag_vocab"]),
        num_numeric_features=len(checkpoint["num_cols"]),
        num_bool_features=len(checkpoint["bool_cols"]),
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, checkpoint


def predict_boundaries(page_df, model, tokenizer, checkpoint, device):
    """
    Run boundary detection on a single page dataframe.
    Return sorted list of predicted start indices.
    """
    cfg = checkpoint["cfg"]

    loader = make_loader(
        page_df, tokenizer,
        checkpoint["tag_vocab"],
        checkpoint["parent_tag_vocab"],
        checkpoint["num_cols"],
        checkpoint["bool_cols"],
        checkpoint["num_mean"],
        checkpoint["num_std"],
        batch_size=1,
        max_tokens=cfg["model"]["max_tokens"],
        shuffle=False
    )

    pages = collect_page_probs_and_truth(loader, model, device)

    if len(pages) == 0:
        return []

    prob_B, _ = pages[0]

    return pick_starts_from_probs(
        prob_B,
        threshold=checkpoint["best_th"],
        nms_k=cfg["inference"]["nms_k"],
        min_gap=cfg["inference"]["min_gap"]
    )


if __name__ == "__main__":

    CHECKPOINT = "models/dom_extractor_checkpoint.pt"
    DATA_CSV = "data/full_data.csv"
    SOURCE = "nacacnet.org_pattern_labeled" # swap this to change page

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model, tokenizer, checkpoint = load_checkpoint(CHECKPOINT)
    model = model.to(DEVICE)

    # load page
    df = pd.read_csv(DATA_CSV)
    page_df = df[df["source"] == SOURCE].sort_values("rendering_order").reset_index(drop=True)

    print(f"Running inference on: {SOURCE}")
    print(f"Nodes: {len(page_df)}")

    # predict
    pred_starts = predict_boundaries(page_df, model, tokenizer, checkpoint, DEVICE)

    print(f"Predicted event starts: {pred_starts}")
    print(f"Number of events detected: {len(pred_starts)}")