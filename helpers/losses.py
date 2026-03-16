import torch
import torch.nn as nn
from collections import Counter


def bio_loss(device):
    """simple unweighted loss"""
    return nn.CrossEntropyLoss(ignore_index=-100).to(device) # skip padding


def weighted_bio_loss(train_df, device, weight_cap=50.0):
    """weighted loss - better if model is baad with B/I"""
    bio_counts = Counter(train_df["bio"].tolist())
    bio_total = sum(bio_counts.get(i, 0) for i in [0, 1, 2])
    
    bio_w = []
    for i in [0, 1, 2]:
        c = bio_counts.get(i, 1)
        bio_w.append(bio_total / c)
        
    bio_w = torch.tensor(bio_w, dtype=torch.float32, device=device)
    bio_w = torch.clamp(bio_w, max=weight_cap)
    
    return nn.CrossEntropyLoss(weight=bio_w, ignore_index=-100)

def csel_loss(embeddings, event_ids, device, margin=1.0, lambda_push=0.5):
    """
    Cross-Structure Equivariance Loss (CSEL).

    Pulls together embeddings of nodes that occupy the same structural
    position (role) across different event instances on the same page.
    Pushes apart embeddings of nodes at different positions within events.

    Args:
        embeddings : (N, d) tensor — transformer outputs for all nodes on page
        event_ids  : list/array of length N — event_id per node, None/NaN for O-nodes
        device     : torch device
        margin     : push margin (default 1.0)
        lambda_push: weight on the push term (default 0.5)

    Returns:
        scalar tensor
    """
    import torch
    import torch.nn.functional as F
    import math

    # --- build role -> [node indices] mapping ---
    role_to_indices = {}   # {role_position: [idx, idx, ...]}
    pos_counter = {}       # {event_id: current_position}

    for i, eid in enumerate(event_ids):
        # treat NaN / None / 0-float as O-node
        if eid is None:
            continue
        try:
            if math.isnan(float(eid)):
                continue
        except (TypeError, ValueError):
            continue

        eid_key = float(eid)
        pos = pos_counter.get(eid_key, 0)
        role_to_indices.setdefault(pos, []).append(i)
        pos_counter[eid_key] = pos + 1

    # need at least 2 roles with at least 2 nodes each to compute anything
    valid_roles = {r: idxs for r, idxs in role_to_indices.items() if len(idxs) >= 2}
    if len(valid_roles) < 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    pull = torch.tensor(0.0, device=device)
    push = torch.tensor(0.0, device=device)
    n_pull, n_push = 0, 0

    role_list = sorted(valid_roles.keys())

    # --- pull: same role, different events -> embeddings should be close ---
    for role, idxs in valid_roles.items():
        vecs = embeddings[idxs]                     # (k, d)
        # use mean prototype rather than all O(k²) pairs — efficient & stable
        prototype = vecs.mean(dim=0, keepdim=True)  # (1, d)
        dists = F.mse_loss(vecs, prototype.expand_as(vecs), reduction='sum')
        pull = pull + dists
        n_pull += len(idxs)

    # --- push: adjacent roles -> embeddings should be far apart ---
    for i in range(len(role_list) - 1):
        r_a, r_b = role_list[i], role_list[i + 1]
        idxs_a = valid_roles[r_a]
        idxs_b = valid_roles[r_b]
        # sample up to 4 pairs to keep cost O(1) per role boundary
        sample_a = idxs_a[:4]
        sample_b = idxs_b[:4]
        for ia in sample_a:
            for ib in sample_b:
                dist = F.pairwise_distance(
                    embeddings[ia].unsqueeze(0),
                    embeddings[ib].unsqueeze(0)
                )
                push = push + F.relu(margin - dist)
                n_push += 1

    if n_pull > 0:
        pull = pull / n_pull
    if n_push > 0:
        push = push / n_push

    return pull + lambda_push * push