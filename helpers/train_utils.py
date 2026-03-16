import torch
from functools import partial
from torch.utils.data import DataLoader
from helpers.dataset import PageDataset, combine_pages
from helpers.losses import csel_loss


# Data Loader ------------------------------------------------
def make_loader(df, tokenizer,
                 tag_vocab, parent_tag_vocab,
                 num_cols, bool_cols,
                 num_mean, num_std,
                 batch_size=2, max_tokens=64, 
                 shuffle=True, num_workers=0):
    """
    Creates a single DataLoader for the given dataframe.
    shuffle=True for training, False for val/test/inference.
    """
    dataset = PageDataset(
        df, tokenizer=tokenizer,
        tag_vocab=tag_vocab, parent_tag_vocab=parent_tag_vocab,
        num_cols=num_cols, bool_cols=bool_cols,
        mean=num_mean, std=num_std,
        max_tokens=max_tokens
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(combine_pages, tokenizer=tokenizer),
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )


# Single Training/Eval Epoch ------------------------------------------
def run_epoch(model, optimizer, loader,
              loss_fn, device,
              training=True,
              lambda_csel=0.0):
    """
    Runs one full pass through the loader.
    Returns average loss per batch.
 
    Args:
        lambda_csel : weight for the CSEL regularizer.
                      0.0 (default) disables it entirely — safe for eval.
                      Recommended starting value: 0.1
    """
    model.train() if training else model.eval()
    total_loss = 0.0

    for batch in loader:
        enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
        node_mask = batch["node_mask"].to(device).bool()
        bio_y = batch["bio_y"].to(device)

        with torch.set_grad_enabled(training):

            # --- forward pass ---
            # model now returns (logits, embeddings); _ discards embeddings
            # when CSEL is off so there's no overhead in eval paths.
            bio_logits, node_embeddings = model(
                enc=enc,
                node_offsets=batch["node_offsets"],
                node_mask=node_mask,
                tag_id=batch["tag_id"].to(device),
                parent_tag_id=batch["parent_tag_id"].to(device),
                num_feats=batch["num_feats"].to(device),
                bool_feats=batch["bool_feats"].to(device),
            )
 
            # --- BIO cross-entropy (unchanged) ---
            loss = loss_fn(bio_logits.view(-1, 3), bio_y.view(-1))
 
            # --- CSEL regularizer (training only, when enabled) ---
            if training and lambda_csel > 0.0:
                # batch_size is always 1, so we process the single page.
                # We strip padding using node_mask so csel_loss only sees
                # real nodes — no -100 sentinels, no padded zeros.
                B = node_mask.size(0)
                csel_total = torch.tensor(0.0, device=device)
                n_pages = 0
 
                for b in range(B):
                    valid_idx = torch.where(node_mask[b])[0]   # real node positions
                    if valid_idx.numel() < 2:
                        continue
 
                    # embeddings for real nodes on this page: (N_valid, d_model)
                    page_embeddings = node_embeddings[b, valid_idx]
 
                    # BIO labels for real nodes — guaranteed no -100 here
                    page_bio_y = bio_y[b, valid_idx]
 
                    # skip pages with fewer than 2 B-nodes (nothing to contrast)
                    if (page_bio_y == 1).sum() < 2:
                        continue
 
                    csel_total = csel_total + csel_loss(
                        page_embeddings, page_bio_y, device
                    )
                    n_pages += 1
 
                if n_pages > 0:
                    loss = loss + lambda_csel * (csel_total / n_pages)
 
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.detach().item()

    return total_loss / max(1, len(loader))