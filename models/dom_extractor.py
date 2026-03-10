import torch
import torch.nn as nn
from transformers import AutoModel


# DOM Extractor Model -------------------------------------------------

class DOMAwareEventExtractor(nn.Module):
    """actual model (Text encoder + DOM Transformer"""
    def __init__(
        self,
        text_model_name: str,
        num_field_labels: int,
        tag_vocab_size: int,
        parent_tag_vocab_size: int,
        num_numeric_features: int,
        num_bool_features: int,
        d_model: int = 128,  # model dimension
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        # text encoder part (DistilBERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size

        self.text_proj = nn.Linear(text_dim, d_model)

        # node embeddings
        self.tag_emb = nn.Embedding(tag_vocab_size, d_model)
        self.parent_tag_emb = nn.Embedding(parent_tag_vocab_size, d_model)

        self.num_proj = nn.Linear(num_numeric_features, d_model)
        self.bool_proj = nn.Linear(num_bool_features, d_model)

        self.layernorm = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True  # DOM transfromer part
        )
        self.node_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # predicrion heads
        self.field_head = nn.Linear(d_model, num_field_labels)  # field classification
        self.bio_head = nn.Linear(d_model, 3)       # O/B/I
        self.in_event_head = nn.Linear(d_model, 1)  # binary

    def forward(
        self,
        enc,
        node_offsets,
        node_mask,
        tag_id,
        parent_tag_id,
        num_feats,
        bool_feats
    ):
        # encode text
        out = self.text_encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]     # [total_nodes, text_dim]
        node_text = self.text_proj(cls)          # [total_nodes, d_model]

        B, max_nodes = node_mask.shape
        packed = node_text.new_zeros((B, max_nodes, node_text.shape[-1]))

        for i, (s, e) in enumerate(node_offsets):
            packed[i, : (e - s), :] = node_text[s:e]

        # add other feature info
        x = (
            packed
            + self.tag_emb(tag_id)
            + self.parent_tag_emb(parent_tag_id)
            + self.num_proj(num_feats)
            + self.bool_proj(bool_feats)
        )
        x = self.layernorm(x)

        # transformer
        key_padding_mask = ~node_mask
        x = self.node_encoder(x, src_key_padding_mask=key_padding_mask)

        # predictions per node
        field_logits = self.field_head(x)                   # field label
        bio_logits = self.bio_head(x)                       # BIO tag
        in_event_logits = self.in_event_head(x).squeeze(-1) # is event?

        return field_logits, bio_logits, in_event_logits
    

# Model Initialization/Optimization ------------------------------------

def init_model_and_optim(
    text_model_name,
    num_field_labels,
    tag_vocab_size, parent_tag_vocab_size,
    num_numeric_features, num_bool_features,
    device,
    lr_bert=5e-6, lr_other=1e-4, weight_decay=0.01
):
    model = DOMAwareEventExtractor(
        text_model_name=text_model_name,
        num_field_labels=num_field_labels,
        tag_vocab_size=tag_vocab_size,
        parent_tag_vocab_size=parent_tag_vocab_size,
        num_numeric_features=num_numeric_features,
        num_bool_features=num_bool_features,
        d_model=128,
        nhead=4,
        num_layers=2
    ).to(device)

    bert_params = []
    other_params = []

    for n, p in model.named_parameters():
        if n.startswith("text_encoder."):
            bert_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": lr_bert, "weight_decay": weight_decay},
            {"params": other_params, "lr": lr_other, "weight_decay": weight_decay},
        ]
    )

    return model, optimizer


def set_bert_trainable(model, trainable: bool):
    for p in model.text_encoder.parameters():
        p.requires_grad = trainable