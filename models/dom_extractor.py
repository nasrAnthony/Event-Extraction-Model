import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# DOM Extractor Model -------------------------------------------------

class DOMAwareEventExtractor(nn.Module):
    """
    Text encoder + DOM Transformer + Anchor Prototype Attention (APA).

    APA is a lightweight module inserted between the DOM transformer and the
    BIO classification head.  After the transformer produces contextual node
    embeddings, APA computes a single page-level anchor prototype by attending
    softly over all nodes with a learned query vector.  It then augments each
    node's embedding with its cosine similarity to that prototype before the
    BIO head makes its prediction.

    This gives the BIO head an explicit, per-node signal — "how much does this
    node resemble what the model currently thinks an anchor looks like on this
    page?" — computed in one forward pass with no recurrence and no dependency
    on previous predictions.  The prototype is trained end-to-end through the
    BIO loss: the model learns that a useful prototype is one that makes correct
    boundary predictions easy.

    New parameters (d_model=64):
        apa_query      : Linear(d_model, 1, bias=False)  —  64 params
        apa_proj       : Linear(d_model + 1, d_model)    —  4 160 params
        apa_norm       : LayerNorm(d_model)               —  128 params
    Total: ~4 352 additional parameters — negligible relative to full model.
    """
    def __init__(
        self,
        text_model_name: str,
        tag_vocab_size: int,
        parent_tag_vocab_size: int,
        num_numeric_features: int,
        num_bool_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_tag=True,
        use_parent_tag=True,
        text_drop_rate: float = 0.0
    ):
        super().__init__()
        self.text_drop_rate = text_drop_rate

        # text encoder (DistilBERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, d_model)

        self.tag_emb = nn.Embedding(tag_vocab_size, d_model)
        self.parent_tag_emb = nn.Embedding(parent_tag_vocab_size, d_model)
        self.use_tag = use_tag
        self.use_parent_tag = use_parent_tag

        # structural feature projections
        self.num_proj  = nn.Linear(num_numeric_features, d_model)
        self.bool_proj = nn.Linear(num_bool_features, d_model)

        self.layernorm = nn.LayerNorm(d_model)

        # DOM transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.node_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ── Anchor Prototype Attention (APA) ──────────────────────────────
        # apa_query   : learned vector that attends over node embeddings to
        #               form the page-level anchor prototype.
        # apa_proj    : projects [h_i ‖ cos_sim(h_i, μ)] back to d_model,
        #               fusing the prototype-similarity signal into each node.
        # apa_norm    : post-fusion layer norm with residual connection.
        self.apa_query = nn.Linear(d_model, 1, bias=False)
        self.apa_proj  = nn.Linear(d_model + 1, d_model)
        self.apa_norm  = nn.LayerNorm(d_model)
        # -----------------------------------------------------------------

        # BIO prediction head — operates on APA-augmented embeddings
        self.bio_head = nn.Linear(d_model, 3)  # O / B / I

    def forward(self, enc, node_offsets, node_mask, tag_id, parent_tag_id, num_feats, bool_feats):
        # ── 1. Encode each node's text ────────────────────────────────────
        out      = self.text_encoder(**enc)
        cls      = out.last_hidden_state[:, 0, :]   # [total_nodes, text_dim]
        node_text = self.text_proj(cls)             # [total_nodes, d_model]

        if self.training and self.text_drop_rate > 0:
            drop_mask = (torch.rand(node_text.size(0), 1, device=node_text.device) > self.text_drop_rate).float()
            node_text = node_text * drop_mask

        # ── 2. Re-pack flat nodes → [B, max_nodes, d_model] ──────────────
        B, max_nodes = node_mask.shape
        packed = node_text.new_zeros((B, max_nodes, node_text.shape[-1]))
        for i, (s, e) in enumerate(node_offsets):
            packed[i, : (e - s), :] = node_text[s:e]

        # ── 3. Combine all node features ──────────────────────────────────
        x = (
            packed
            + self.num_proj(num_feats)
            + self.bool_proj(bool_feats)
        )
        if self.use_tag:
            x = x + self.tag_emb(tag_id)
        if self.use_parent_tag:
            x = x + self.parent_tag_emb(parent_tag_id)

        x = self.layernorm(x)

        # ── 4. DOM transformer ────────────────────────────────────────────
        key_padding_mask = ~node_mask          # True = ignore (padding)
        x = self.node_encoder(x, src_key_padding_mask=key_padding_mask)
        # x : [B, max_nodes, d_model] — contextual node representations

        # ── 5. Anchor Prototype Attention (APA) ───────────────────────────
        #
        # Step A: compute attention logits over all nodes and mask padding
        attn_logits = self.apa_query(x).squeeze(-1)          # [B, max_nodes]
        attn_logits = attn_logits.masked_fill(~node_mask, float('-inf'))

        # Step B: softmax → weighted sum → page-level prototype μ
        attn_weights = torch.softmax(attn_logits, dim=-1)    # [B, max_nodes]
        # [B, max_nodes, 1] * [B, max_nodes, d] → sum → [B, d_model]
        mu = (attn_weights.unsqueeze(-1) * x).sum(dim=1)    # [B, d_model]

        # Step C: per-node cosine similarity to the prototype
        mu_exp  = mu.unsqueeze(1).expand_as(x)              # [B, max_nodes, d_model]
        cos_sim = F.cosine_similarity(x, mu_exp, dim=-1, eps=1e-8)  # [B, max_nodes]

        # Step D: augment each node with its prototype similarity
        # concat [h_i ‖ cos_sim_i], project back to d_model, residual + norm
        cos_feat = cos_sim.unsqueeze(-1)                     # [B, max_nodes, 1]
        x_aug    = self.apa_proj(torch.cat([x, cos_feat], dim=-1))  # [B, max_nodes, d_model]
        x_aug    = self.apa_norm(x_aug + x)                 # residual connection
        # -----------------------------------------------------------------

        # ── 6. BIO head on prototype-aware embeddings ─────────────────────
        logits = self.bio_head(x_aug)                        # [B, max_nodes, 3]

        # Return both so CSEL/TACO in train_utils can operate on x_aug.
        # Eval paths unpack as:  logits, _ = model(...)
        return logits, x_aug


# Model/Optimizer Initialization ------------------------------------

def init_model_and_optim(cfg, tag_vocab_size, parent_tag_vocab_size,
    num_numeric_features, num_bool_features, device):
    """
    Make and return an instance of model + optimizer (using config values).
    No new arguments needed — APA layers are always instantiated.
    """
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]

    model = DOMAwareEventExtractor(
        text_model_name=model_cfg["name"],
        tag_vocab_size=tag_vocab_size,
        parent_tag_vocab_size=parent_tag_vocab_size,
        num_numeric_features=num_numeric_features,
        num_bool_features=num_bool_features,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        use_tag=model_cfg.get("use_tag", True),
        use_parent_tag=model_cfg.get("use_parent_tag", True),
        text_drop_rate=model_cfg.get("text_drop_rate", 0.0)
    ).to(device)

    bert_params  = []
    other_params = []
    for n, p in model.named_parameters():
        if n.startswith("text_encoder."):
            bert_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": bert_params,  "lr": float(training_cfg["lr_bert"]),
         "weight_decay": float(training_cfg["weight_decay"])},
        {"params": other_params, "lr": float(training_cfg["lr_other"]),
         "weight_decay": float(training_cfg["weight_decay"])},
    ])

    return model, optimizer


def set_bert_trainable(model, trainable: bool):
    """Freeze / unfreeze DistilBERT parameters."""
    for p in model.text_encoder.parameters():
        p.requires_grad = trainable
