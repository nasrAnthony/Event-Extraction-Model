import torch
import torch.nn as nn
from transformers import AutoModel


# DOM Extractor Model -------------------------------------------------

class DOMAwareEventExtractor(nn.Module):
    """actual model (Text encoder + DOM Transformer)"""
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
        use_parent_tag=True
    ):
        super().__init__()

        # text encoder part (DistilBERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, d_model)
        
        self.tag_emb = nn.Embedding(tag_vocab_size, d_model)
        self.parent_tag_emb = nn.Embedding(parent_tag_vocab_size, d_model)
        self.use_tag = use_tag
        self.use_parent_tag = use_parent_tag

        # node embeddings
        self.num_proj = nn.Linear(num_numeric_features, d_model)
        self.bool_proj = nn.Linear(num_bool_features, d_model)

        self.layernorm = nn.LayerNorm(d_model)
        
        # DOM transfromer part
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True  
        )
        self.node_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # BIO prediction head
        self.bio_head = nn.Linear(d_model, 3) # O / B / I

    def forward(self, enc, node_offsets, node_mask, tag_id, parent_tag_id, num_feats, bool_feats):
        # encode each node's text
        out = self.text_encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]     # [total_nodes, text_dim]
        node_text = self.text_proj(cls)          # [total_nodes, d_model]

        # move back from flat nodes to [B, max_nodes, d_model]
        B, max_nodes = node_mask.shape
        packed = node_text.new_zeros((B, max_nodes, node_text.shape[-1]))
        for i, (s, e) in enumerate(node_offsets):
            packed[i, : (e - s), :] = node_text[s:e]

        # combine all node features
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

        # run through DOM transformer
        key_padding_mask = ~node_mask
        x = self.node_encoder(x, src_key_padding_mask=key_padding_mask)

        # predict BIO logits per node
        return self.bio_head(x) # [B, max_nodes, 3]
    

# Model/Optimizer Initialization ------------------------------------

def init_model_and_optim(cfg, tag_vocab_size, parent_tag_vocab_size,
    num_numeric_features, num_bool_features, device):
    """
    make and return instance of model/optimizar (using config values)
    """
    model_cfg = cfg["model"]
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
        use_parent_tag=model_cfg.get("use_parent_tag", True)
    ).to(device)

    bert_params = []
    other_params = []
    for n, p in model.named_parameters():
        if n.startswith("text_encoder."):
            bert_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
            {"params": bert_params, "lr": float(training_cfg["lr_bert"]), "weight_decay": float(training_cfg["weight_decay"])},
            {"params": other_params, "lr": float(training_cfg["lr_other"]), "weight_decay": float(training_cfg["weight_decay"])},
        ])

    return model, optimizer


def set_bert_trainable(model, trainable: bool):
    """freeze/unfreeze DistilBERT parameters (F -> Freeze, T -> Unfreeze)"""
    for p in model.text_encoder.parameters():
        p.requires_grad = trainable