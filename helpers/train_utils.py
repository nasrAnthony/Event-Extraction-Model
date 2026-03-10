import torch
from functools import partial
from torch.utils.data import DataLoader
from helpers.dataset import PageDataset, combine_pages


# Data Loader ------------------------------------------------
def make_loader(df, tokenizer,
                 tag_vocab, parent_tag_vocab,
                 num_cols, bool_cols,
                 num_mean, num_std,
                 batch_size=2, max_tokens=64, 
                 shuffle=False, num_workers=0):
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
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )


# Single Training/Eval Epoch ------------------------------------------
def run_epoch(model, optimizer, loader,
              loss_fn, device,
              training=True):
    """
    Runs one full pass through the loader.
    Returns average loss per batch.
    """
    model.train() if training else model.eval()
    total_loss = 0.0

    for batch in loader:
        enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
        node_mask = batch["node_mask"].to(device).bool()
        bio_y = batch["bio_y"].to(device)

        with torch.set_grad_enabled(training):
            bio_logits = model(
                enc=enc,
                node_offsets=batch["node_offsets"],
                node_mask=node_mask,
                tag_id=batch["tag_id"].to(device),
                parent_tag_id=batch["parent_tag_id"].to(device),
                num_feats=batch["num_feats"].to(device),
                bool_feats=batch["bool_feats"].to(device),
            )

            loss = loss_fn(bio_logits.view(-1, 3), bio_y.view(-1))

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.detach().item()

    return total_loss / max(1, len(loader))