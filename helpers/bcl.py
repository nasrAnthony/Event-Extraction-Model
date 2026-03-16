import torch
import torch.nn.functional as F


def boundary_contrast_loss(
    embeddings, bio_labels, node_mask, device,
    temperature=0.1, neighbor_radius=1,
):
    """
    Boundary Contrast Loss (InfoNCE variant).
    
    Pulls B-tag embeddings together, pushes them away from their
    ±neighbor_radius neighbors (the nodes that cause ±1 overshoot).
    """
    batch_size = embeddings.size(0)
    total_loss = torch.tensor(0.0, device=device)
    n_anchors = 0

    for b in range(batch_size):
        valid = node_mask[b]
        if valid.sum() < 3:
            continue

        emb = embeddings[b][valid]
        labels = bio_labels[b][valid]
        N = emb.size(0)

        b_indices = torch.where(labels == 1)[0]
        if len(b_indices) < 2:
            continue

        # hard negatives: neighbors of B-tags that aren't B-tags themselves
        b_set = set(b_indices.tolist())
        neg_set = set()
        for bi in b_indices:
            bi_int = bi.item()
            for offset in range(-neighbor_radius, neighbor_radius + 1):
                ni = bi_int + offset
                if ni != bi_int and 0 <= ni < N and ni not in b_set:
                    neg_set.add(ni)

        if not neg_set:
            continue

        neg_indices = torch.tensor(sorted(neg_set), device=device)

        # normalize embeddings
        b_embs_norm = F.normalize(emb[b_indices], dim=-1)
        neg_embs_norm = F.normalize(emb[neg_indices], dim=-1)

        for i in range(len(b_indices)):
            anchor = b_embs_norm[i]

            # positives: other B-tags
            pos_mask = torch.ones(len(b_indices), dtype=torch.bool, device=device)
            pos_mask[i] = False
            pos_sims = (anchor * b_embs_norm[pos_mask]).sum(dim=-1) / temperature
            neg_sims = (anchor * neg_embs_norm).sum(dim=-1) / temperature

            # InfoNCE: -log( sum(exp(pos)) / sum(exp(all)) )
            all_sims = torch.cat([pos_sims, neg_sims])
            total_loss = total_loss + (torch.logsumexp(all_sims, dim=0) - torch.logsumexp(pos_sims, dim=0))
            n_anchors += 1

    if n_anchors > 0:
        total_loss = total_loss / n_anchors

    return total_loss


def reconstruct_event_ids_from_bio(bio_labels):
    """
    Convert BIO label tensor → event_id list for CSEL.
    
    B starts a new event, I continues it, O becomes None.
    Returns list of float/None suitable for csel_loss().
    """
    event_ids = []
    current_event = 0
    in_event = False

    for label in bio_labels.tolist():
        if label == 1:  # B
            current_event += 1
            event_ids.append(float(current_event))
            in_event = True
        elif label == 2 and in_event:  # I
            event_ids.append(float(current_event))
        else:  # O
            event_ids.append(None)
            in_event = False

    return event_ids