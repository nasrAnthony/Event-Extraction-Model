import numpy as np
import torch
from sklearn.metrics import f1_score


# Peak-based decoding + start metrics (NMS/min-gap/tolerance) ---------------
def pick_starts_from_probs(probs, threshold=0.5, nms_k=1, min_gap=2):
    """
    probs: 1D numpy array of prob(B) per node (valid nodes only)
    Returns sorted list of predicted start indices (within valid-node indexing)
    Steps:
      - threshold
      - local maxima (nms_k neighborhood)
      - greedy keep with min_gap
    """
    probs = np.asarray(probs)
    N = probs.shape[0]
    if N == 0:
        return []

    # Candidates above threshold
    cand = np.where(probs >= threshold)[0]
    if cand.size == 0:
        return []

    # Local maxima within +/- nms_k
    if nms_k > 0:
        keep = []
        for i in cand:
            lo = max(0, i - nms_k)
            hi = min(N, i + nms_k + 1)
            if probs[i] >= probs[lo:hi].max() - 1e-12:
                keep.append(i)
        cand = np.array(keep, dtype=int)
        if cand.size == 0:
            return []

    # Greedy by prob descending, enforce min_gap
    order = cand[np.argsort(-probs[cand])]
    chosen = []
    for idx in order:
        if all(abs(idx - j) > min_gap for j in chosen):
            chosen.append(int(idx))

    return sorted(chosen)


def start_prf_with_tolerance(true_starts, pred_starts, tol=1):
    """
    true_starts, pred_starts: sorted lists of indices
    tolerance: pred counts as TP if within +/- tol of an unmatched true
    """
    true_starts = list(true_starts)
    pred_starts = list(pred_starts)

    matched_true = set()
    tp = 0

    for p in pred_starts:
        best = None
        best_dist = None

        for ti, t in enumerate(true_starts):
            if ti in matched_true:
                continue

            d = abs(p - t)
            if d <= tol and (best_dist is None or d < best_dist):
                best = ti
                best_dist = d

        if best is not None:
            matched_true.add(best)
            tp += 1

    fp = len(pred_starts) - tp
    fn = len(true_starts) - tp

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    return prec, rec, f1


# Probabilities per Page -----------------------------------------------------
@torch.no_grad()
def collect_page_probs_and_truth(loader, model, device):
    """
    Returns list of (prob_B_valid, true_start_valid) for each page instance in loader.
    prob_B_valid: np array length = #valid nodes
    true_start_valid: np array length = #valid nodes, {0,1} where start_event==1 (bio==B)
    """
    model.eval()
    out = []

    for batch in loader:
        enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
        node_mask = batch["node_mask"].to(device).bool()

        field_logits, bio_logits, in_event_logits = model(
            enc=enc,
            node_offsets=batch["node_offsets"],
            node_mask=node_mask,
            tag_id=batch["tag_id"].to(device),
            parent_tag_id=batch["parent_tag_id"].to(device),
            num_feats=batch["num_feats"].to(device),
            bool_feats=batch["bool_feats"].to(device),
        )

        # prob of B (bio label 1)
        prob_B = torch.softmax(bio_logits, dim=-1)[..., 1]  # [B,N]

        bio_y = batch["bio_y"].to(device)
        true_start = (bio_y == 1).long()

        prob_B = prob_B.detach().cpu()
        true_start = true_start.detach().cpu()
        mask = node_mask.detach().cpu()

        B, N = prob_B.shape

        for b in range(B):
            valid = torch.where(mask[b])[0]
            if valid.numel() == 0:
                continue

            out.append((
                prob_B[b, valid].numpy(),
                true_start[b, valid].numpy().astype(int)
            ))

    return out


# Threshold search -----------------------------------------------------
@torch.no_grad()
def find_best_threshold_peak(
    loader,
    model,
    device,
    thresholds=None,
    nms_k=1,
    min_gap=2,
    tol=1
):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    pages = collect_page_probs_and_truth(loader, model, device)

    if len(pages) == 0:
        return 0.5, 0.0

    best_th = 0.5
    best_f1 = -1.0

    for th in thresholds:
        TP = FP = FN = 0

        for prob_B, true_start in pages:
            true_idx = np.where(true_start == 1)[0].tolist()
            pred_idx = pick_starts_from_probs(
                prob_B,
                threshold=th,
                nms_k=nms_k,
                min_gap=min_gap
            )

            matched_true = set()
            tp = 0

            for pr in pred_idx:
                best = None
                best_dist = None

                for ti, t in enumerate(true_idx):
                    if ti in matched_true:
                        continue

                    d = abs(pr - t)
                    if d <= tol and (best_dist is None or d < best_dist):
                        best = ti
                        best_dist = d

                if best is not None:
                    matched_true.add(best)
                    tp += 1

            fp = len(pred_idx) - tp
            fn = len(true_idx) - tp

            TP += tp
            FP += fp
            FN += fn

        prec = TP / (TP + FP + 1e-9)
        rec = TP / (TP + FN + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)

    return best_th, float(best_f1)


# Boundary metrics --------------------------------------------------------
@torch.no_grad()
def boundary_metrics_peak(
    loader,
    model,
    device,
    threshold,
    nms_k=1,
    min_gap=2,
    tol=1
):
    pages = collect_page_probs_and_truth(loader, model, device)

    if len(pages) == 0:
        return 0.0, 0.0, 0.0

    TP = FP = FN = 0

    for prob_B, true_start in pages:
        true_idx = np.where(true_start == 1)[0].tolist()
        pred_idx = pick_starts_from_probs(
            prob_B,
            threshold=threshold,
            nms_k=nms_k,
            min_gap=min_gap
        )

        matched_true = set()
        tp = 0

        for pr in pred_idx:
            best = None
            best_dist = None

            for ti, t in enumerate(true_idx):
                if ti in matched_true:
                    continue

                d = abs(pr - t)
                if d <= tol and (best_dist is None or d < best_dist):
                    best = ti
                    best_dist = d

            if best is not None:
                matched_true.add(best)
                tp += 1

        fp = len(pred_idx) - tp
        fn = len(true_idx) - tp

        TP += tp
        FP += fp
        FN += fn

    prec = TP / (TP + FP + 1e-9)
    rec = TP / (TP + FN + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    return float(prec), float(rec), float(f1)


# Field metrics --------------------------------------------------------
@torch.no_grad()
def field_metrics_fast(
    loader,
    model,
    device,
    label2id,
    average="micro",
    restrict_to_true_in_event=False
):
    model.eval()

    yt = []
    yp = []

    OTHER_ID = label2id["Other"]

    for batch in loader:
        enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
        node_mask = batch["node_mask"].to(device).bool()

        field_y = batch["field_y"].to(device)
        in_event_y = batch["in_event_y"].to(device)

        field_logits, bio_logits, in_event_logits = model(
            enc=enc,
            node_offsets=batch["node_offsets"],
            node_mask=node_mask,
            tag_id=batch["tag_id"].to(device),
            parent_tag_id=batch["parent_tag_id"].to(device),
            num_feats=batch["num_feats"].to(device),
            bool_feats=batch["bool_feats"].to(device),
        )

        pred = torch.argmax(field_logits, dim=-1)

        valid = node_mask & (field_y != -100) & (field_y != OTHER_ID)

        if restrict_to_true_in_event:
            valid = valid & (in_event_y == 1)

        yt.extend(field_y[valid].detach().cpu().tolist())
        yp.extend(pred[valid].detach().cpu().tolist())

    if len(yt) == 0:
        return 0.0

    return float(f1_score(yt, yp, average=average, zero_division=0))