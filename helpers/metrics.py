import numpy as np
import torch
from sklearn.metrics import f1_score


# Get boundary probabilities per page -----------------------------------------------------
@torch.no_grad()
def collect_page_probs_and_truth(loader, model, device):
    """
    run model over all pages in loader
    return list of (probs_full, prob_B, true_start), one per page
    """
    model.eval()
    out = []
    
    with torch.no_grad():
        for batch in loader:
            enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
            node_mask = batch["node_mask"].to(device)

            # run model and return bio logits for every node in every page
            bio_logits = model(
                enc=enc,
                node_offsets=batch["node_offsets"],
                node_mask = node_mask,
                tag_id=batch["tag_id"].to(device),
                parent_tag_id=batch["parent_tag_id"].to(device),
                num_feats=batch["num_feats"].to(device),
                bool_feats=batch["bool_feats"].to(device),
            )

            # softmax over all classes and take B column
            probs = torch.softmax(bio_logits, dim=-1)  # [B, N, 3]
            bio_y = batch["bio_y"].to(device)
            true_start = (bio_y == 1).long()

            probs = probs.detach().cpu()
            true_start = true_start.detach().cpu()
            mask = node_mask.detach().cpu() # move back to cpu

            # strip padding
            for b in range(len(batch["node_offsets"])):
                valid = torch.where(mask[b])[0] # indicies of real nodes
                if valid.numel() == 0:
                    continue
                probs_valid = probs[b, valid].numpy() # [N_valid, 3]
                prob_B = probs_valid[:, 1].numpy()
                true_valid = true_start[b, valid].numpy().astype(int)
                out.append((probs_valid, prob_B, true_valid))

    return out


# Peak-based decoding + start metrics (NMS/min-gap/tolerance) ---------------
def pick_starts_from_probs(probs, threshold=0.5, nms_k=1, min_gap=2):
    """
    Pick the leftmost local maximum in each contiguous above-threshold region,
    then enforce min_gap left-to-right.
    """
    probs = np.asarray(probs)
    N = probs.shape[0]
    if N == 0:
        return []

    # get all candidates above threshold
    above = probs >= threshold
    if not np.any(above):
        return []
    
    # get actual above-threshold regions    
    starts = np.where(above & ~np.r_[False, above[:-1]])[0]
    ends   = np.where(above & ~np.r_[above[1:], False])[0]

    picks = []
    for l, r in zip(starts, ends):
        cand = np.arange(l, r + 1)
        
        # filter to local maxima within nms_k neighborhood
        if nms_k > 0:
            keep = []
            for i in cand:
                lo = max(0, i - nms_k)
                hi = min(N, i + nms_k + 1)
                if probs[i] >= probs[lo:hi].max() - 1e-12:
                    keep.append(i)
            cand = np.array(keep, dtype=int)
            if cand.size == 0:
                continue
        
        # pick leftmost surviving candidate
        picks.append(int(cand.min()))

    # enforce left-to-right with a min_gap between event starts
    chosen = []
    for idx in sorted(picks):
        if not chosen or (idx - chosen[-1]) > min_gap:
            chosen.append(idx)

    return chosen


# TP/FP/FN counts for a single page ------------------------------------------
def start_prf_with_tolerance(true_starts, pred_starts, tol=1):
    """
    Return (tp, fp, fn) counts for a single page.
    A prediction counts as TP if within +/-tol of an unmatched true start.
    """
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

    return tp, fp, fn

def compute_prf(tp, fp, fn):
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return float(prec), float(rec), float(f1)


# Threshold search -----------------------------------------------------
@torch.no_grad()
def find_best_threshold_peak(loader, model, device,
                             thresholds=None, nms_k=1, min_gap=2, tol=1):
    """
    Sweep thresholds to find the one giving best F1 across all pages
    Return (best_threshold, best_f1)
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    pages = collect_page_probs_and_truth(loader, model, device)
    if len(pages) == 0:
        return 0.5, 0.0

    best_th = 0.5
    best_f1 = -1.0

    for th in thresholds:
        TP = FP = FN = 0

        for probs_full, prob_B, true_start in pages:
            true_idx = np.where(true_start == 1)[0].tolist()
            pred_idx = pick_starts_from_probs(prob_B, threshold=th,
                                              nms_k=nms_k, min_gap=min_gap)
            tp, fp, fn = start_prf_with_tolerance(true_idx, pred_idx, tol)
            TP += tp
            FP += fp
            FN += fn

        _, _, f1 = compute_prf(TP, FP, FN)

        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)

    return best_th, float(best_f1)


# Boundary metrics with a fixed threshold -----------------------------------------
@torch.no_grad()
def boundary_metrics_peak(loader, model, device, 
                          threshold, nms_k=1, min_gap=2, tol=1):
    """
    Computes P/R/F1 at a fixed threshold across all pages
    Used after cross val to evaluate the final model with best_th
    """
    pages = collect_page_probs_and_truth(loader, model, device)
    if len(pages) == 0:
        return 0.0, 0.0, 0.0

    TP = FP = FN = 0

    for prob_B, true_start in pages:
        true_idx = np.where(true_start == 1)[0].tolist()
        pred_idx = pick_starts_from_probs(prob_B, threshold=threshold,
                                          nms_k=nms_k, min_gap=min_gap)
        tp, fp, fn = start_prf_with_tolerance(true_idx, pred_idx, tol)
        TP += tp
        FP += fp
        FN += fn

    return compute_prf(TP, FP, FN)

