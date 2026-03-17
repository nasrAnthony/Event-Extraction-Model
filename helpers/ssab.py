"""
ssab.py — Structural Self-Attention Bias

Drop this into your helpers/ directory.

Contains two things:
1. StructuralSelfAttentionBias — computes the bias matrix from DOM features
2. BiasedTransformerEncoder — drop-in replacement for nn.TransformerEncoder
   that accepts and applies the bias matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralSelfAttentionBias(nn.Module):
    """
    Computes a [B, N, N] attention bias matrix from structural DOM features.
    
    For each pair of nodes (i, j), checks structural fingerprint similarity
    at 4 levels:
        Level 1: same tag
        Level 2: same tag + same depth
        Level 3: same tag + same depth + same sibling_index
        Level 4: same tag + same depth + same sibling_index + same parent_tag
    
    Each level has a learned scalar weight (initialized to 0.5).
    The output is added to attention logits before softmax.
    
    Total learnable parameters: 4 scalars.
    """
    
    def __init__(self):
        super().__init__()
        self.bias_weights = nn.Parameter(torch.ones(4) * 0.5)
    
    def forward(self, tag_id, parent_tag_id, depth, sibling_index, node_mask):
        """
        Args:
            tag_id:         [B, N] long — tag vocabulary indices
            parent_tag_id:  [B, N] long — parent tag vocabulary indices
            depth:          [B, N] long — raw integer depth values
            sibling_index:  [B, N] long — raw integer sibling index values
            node_mask:      [B, N] bool — True for real nodes
            
        Returns:
            [B, N, N] float — additive attention bias
        """
        # pairwise equality at each level
        tag_eq = (tag_id.unsqueeze(2) == tag_id.unsqueeze(1)).float()
        depth_eq = (depth.unsqueeze(2) == depth.unsqueeze(1)).float()
        sib_eq = (sibling_index.unsqueeze(2) == sibling_index.unsqueeze(1)).float()
        parent_eq = (parent_tag_id.unsqueeze(2) == parent_tag_id.unsqueeze(1)).float()
        
        # cumulative levels
        level2 = tag_eq * depth_eq
        level3 = level2 * sib_eq
        level4 = level3 * parent_eq
        
        # incremental contributions (so weights are independent)
        l1_only = tag_eq - level2
        l2_only = level2 - level3
        l3_only = level3 - level4
        l4_full = level4
        
        # weighted sum
        bias = (self.bias_weights[0] * l1_only +
                self.bias_weights[1] * l2_only +
                self.bias_weights[2] * l3_only +
                self.bias_weights[3] * l4_full)
        
        # mask padding and self-connections
        B, N = tag_id.shape
        pad_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        diag_mask = ~torch.eye(N, device=tag_id.device, dtype=torch.bool).unsqueeze(0)
        bias = bias * pad_mask.float() * diag_mask.float()
        
        return bias


class BiasedTransformerEncoderLayer(nn.Module):
    """
    One transformer encoder layer that accepts an additive attention bias.
    
    Identical to nn.TransformerEncoderLayer except:
    - Manual multi-head attention so we can inject the bias
    - Pre-norm (norm before attention/ff) for stability
    """
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x, attn_bias=None, src_key_padding_mask=None):
        """
        Args:
            x:                    [B, N, d_model]
            attn_bias:            [B, N, N] additive bias for attention scores
            src_key_padding_mask: [B, N] True = IGNORE (padding)
        """
        B, N, d = x.shape
        H = self.nhead
        hd = self.head_dim
        
        # pre-norm self-attention
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(B, N, H, hd).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, N, H, hd).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, N, H, hd).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (hd ** 0.5)  # [B, H, N, N]
        
        # THIS IS WHERE SSAB HAPPENS: add structural bias to attention scores
        if attn_bias is not None:
            # apply structural bias to first half of heads only, but stronger
            head_mask = torch.zeros(1, self.nhead, 1, 1, device=scores.device)
            head_mask[0, :self.nhead // 2] = 1.0
            scores = scores + attn_bias.unsqueeze(1) * head_mask * 2.0
        
        if src_key_padding_mask is not None:
            scores = scores.masked_fill(
                src_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop1(attn)
        
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, d)
        out = self.out_proj(out)
        x = x + out
        
        # pre-norm feedforward
        x = x + self.drop2(self.ff(self.norm2(x)))
        
        return x


class BiasedTransformerEncoder(nn.Module):
    """
    Stack of BiasedTransformerEncoderLayers.
    Drop-in replacement for nn.TransformerEncoder.
    """
    
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BiasedTransformerEncoderLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, attn_bias=None, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, src_key_padding_mask=src_key_padding_mask)
        return x
