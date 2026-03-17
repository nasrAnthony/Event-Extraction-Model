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