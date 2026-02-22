import torch
import torch.nn as nn
from collections import Counter


def make_losses_for_train_df(
    train_df,
    LABELS,
    device,
    other_scale=0.05,
    weight_cap=50.0
):
    """Weighted losses for the 3 tasks that are being run"""

    # ---- Field weights ----
    label_counts = Counter(train_df["label"].tolist())
    total = sum(label_counts.values())

    weights = []
    for label in LABELS:
        c = label_counts.get(label, 1)
        weights.append(total / c)

    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = torch.clamp(weights, max=weight_cap)

    if "Other" in LABELS:
        weights[LABELS.index("Other")] *= other_scale

    field_loss_fn = nn.CrossEntropyLoss(
        weight=weights,
        ignore_index=-100
    )

    # ---- BIO weights ----
    bio_counts = Counter(train_df["bio"].tolist())

    # bio labels are 0/1/2, but ensure all exist
    bio_total = sum(bio_counts.get(i, 0) for i in [0, 1, 2])

    bio_w = []
    for i in [0, 1, 2]:
        c = bio_counts.get(i, 1)
        bio_w.append(bio_total / c)

    bio_w = torch.tensor(bio_w, dtype=torch.float32, device=device)
    bio_w = torch.clamp(bio_w, max=weight_cap)

    bio_loss_fn = nn.CrossEntropyLoss(
        weight=bio_w,
        ignore_index=-100
    )

    # ---- in_event pos_weight ----
    pos = float(train_df["in_event"].sum())
    neg = float(len(train_df) - pos)

    pos_weight = torch.tensor(
        [neg / (pos + 1e-6)],
        dtype=torch.float32,
        device=device
    )

    in_event_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    return field_loss_fn, bio_loss_fn, in_event_loss_fn