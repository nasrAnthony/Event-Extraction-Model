import torch
from torch.utils.data import DataLoader


# Data Loaders ------------------------------------------------
def make_loaders(
    train_df, val_df,
    tokenizer,
    label2id,
    tag_vocab, parent_tag_vocab,
    num_cols, bool_cols,
    num_mean, num_std,
    batch_size=2, max_tokens=64, num_workers=0
):
    from helpers.dataset import PageDataset, combine_pages

    train_dataset = PageDataset(
        train_df,
        tokenizer=tokenizer,
        label2id=label2id,
        tag_vocab=tag_vocab,
        parent_tag_vocab=parent_tag_vocab,
        num_cols=num_cols,
        bool_cols=bool_cols,
        mean=num_mean,
        std=num_std,
        max_tokens=max_tokens
    )

    val_dataset = PageDataset(
        val_df,
        tokenizer=tokenizer,
        label2id=label2id,
        tag_vocab=tag_vocab,
        parent_tag_vocab=parent_tag_vocab,
        num_cols=num_cols,
        bool_cols=bool_cols,
        mean=num_mean,
        std=num_std,
        max_tokens=max_tokens
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: combine_pages(x, tokenizer),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: combine_pages(x, tokenizer),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader


# Single Training/Eval Epoch ------------------------------------------
def run_epoch(
    model, optimizer, loader,
    field_loss_fn, bio_loss_fn, in_event_loss_fn,
    device,
    w_bio=2.0, w_in_event=1.0,
    training=True
):
    model.train() if training else model.eval()

    total_loss = 0.0

    for batch in loader:
        enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
        node_mask = batch["node_mask"].to(device).bool()

        field_y = batch["field_y"].to(device)
        bio_y = batch["bio_y"].to(device)
        in_event_y = batch["in_event_y"].to(device)

        with torch.set_grad_enabled(training):
            field_logits, bio_logits, in_event_logits = model(
                enc=enc,
                node_offsets=batch["node_offsets"],
                node_mask=node_mask,
                tag_id=batch["tag_id"].to(device),
                parent_tag_id=batch["parent_tag_id"].to(device),
                num_feats=batch["num_feats"].to(device),
                bool_feats=batch["bool_feats"].to(device),
            )

            # Field loss: ALL nodes (including Other), excluding padding
            field_mask = node_mask & (field_y != -100)
            field_loss = field_loss_fn(
                field_logits[field_mask],
                field_y[field_mask]
            )

            # BIO loss: excluding padding
            bio_mask = node_mask & (bio_y != -100)
            bio_loss = bio_loss_fn(
                bio_logits[bio_mask],
                bio_y[bio_mask]
            )

            # in_event loss: excluding padding
            ie_mask = node_mask & (in_event_y != -100)
            in_event_loss = in_event_loss_fn(
                in_event_logits[ie_mask],
                in_event_y[ie_mask].float()
            )

            loss = field_loss + w_bio * bio_loss + w_in_event * in_event_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.detach().item()

    return total_loss / max(1, len(loader))