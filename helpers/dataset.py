import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# Label Merging -------------------------------------------

# map of labels to be merged
LABEL_MERGE_MAP = {
    # keep
    "Other": "Other",

    # Name variants
    "Name": "Name",
    "NameLink": "Name",
    "NameLocation": "Name",

    # Date variants
    "Date": "Date",
    "DateTime": "Date",

    # Time variants
    "Time": "Time",
    "StartTime": "Time",
    "EndTime": "Time",
    "StartEndTime": "Time",
    "TimeLocation": "Time",

    "Location": "Location",

    "Description": "Description",
    "Desc": "Description",
    "Details": "Description",
}

def merge_labels(df, label_col="label", mapping=None, default_to_other=True):
    """
    function to merge similar labels based on mapping
    returns a copy of the input df with merged labels
    """
    df = df.copy()
    mapping = mapping or {}
    df[label_col] = df[label_col].astype(str).map(mapping).fillna("Other")
    return df


def compute_num_stats(train_df: pd.DataFrame, cols):
    """
    mean and standard dev of numerical columns/features. 
    used for normalization/scaling later
    """
    x = train_df[cols].fillna(0).values.astype("float32") # column data organization
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype("float32") # replace very small variance with 1.0
    return mean.astype("float32"), std.astype("float32")


# Dataset Class ------------------------------------------

class PageDataset(Dataset):
    """
    groups rows/nodes by source and tokenizes text 
    applies numeric normalization using provided mean and std
    each dataset item is a whole webpage
    """
    def __init__(self, df, tokenizer, label2id, 
                 tag_vocab, parent_tag_vocab, num_cols, 
                 bool_cols, mean=None, std=None, max_tokens=64):
        self.pages = []
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.tag_vocab = tag_vocab
        self.parent_tag_vocab = parent_tag_vocab
        self.num_cols = num_cols
        self.bool_cols = bool_cols

        # mean and std (empty for safety/normal distribution)
        if mean is None:
            mean = np.zeros((len(num_cols),), dtype="float32")
        if std is None:
            std = np.ones((len(num_cols),), dtype="float32")
        self.mean = mean
        self.std = std

        for src, g in df.groupby("source", sort=False):
            g = g.sort_values("rendering_order").reset_index(drop=True) # extra check

            # TEXT FEATURES: tekenize
            texts = g["text_context"].astype(str).tolist()
            enc = tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=max_tokens,
                return_attention_mask=True,
                return_tensors=None
            )

            # NUMERIC FEATURES: normalize
            num = g[num_cols].fillna(0).values.astype("float32")
            num = (num - self.mean) / self.std

            # overall page dict
            page = {
                "input_ids": enc["input_ids"],  # transformer output
                "attention_mask": enc["attention_mask"],

                "field_y": [label2id[x] for x in g["label"].tolist()],  # field classification
                "bio_y": g["bio"].astype(int).tolist(),  # boundaries
                "in_event_y": g["in_event"].astype(int).tolist(),  # is an event node or not

                "tag_id": [tag_vocab[str(x)] for x in g["tag"]],
                "parent_tag_id": [parent_tag_vocab[str(x)] for x in g["parent_tag"]],

                "num_feats": num,
                "bool_feats": g[bool_cols].astype(int).values.astype("float32")
            }
            self.pages.append(page)

    def __len__(self):
        return len(self.pages) # number of pages

    def __getitem__(self, idx):
        return self.pages[idx] # returns a single page
    

# More Data Manipulation -----------------------------------------------
def combine_pages(batch, tokenizer):
    """
    function deals with flattening text and padding nodes
    batch: list of pages
    returns a usable tensor with all node/page data
    """
    B = len(batch)

    # flattening all text data across all pages [total nodes, token_length]
    flat_text = [
        {"input_ids": ids, "attention_mask": mask}
        for x in batch
        for ids, mask in zip(x["input_ids"], x["attention_mask"])
    ]

    # padding tokens to match max lenght
    enc = tokenizer.pad(flat_text, return_tensors="pt")

    lengths = [len(x["input_ids"]) for x in batch]  # num of nodes per page
    max_nodes = max(lengths)

    offsets = torch.cumsum(torch.tensor([0] + lengths), dim=0)  # to indicate where pages start
    node_offsets = [(offsets[i].item(), offsets[i+1].item()) for i in range(B)]  # page ranges

    # applying pdding
    field_y = pad_sequence([torch.tensor(x["field_y"]) for x in batch], batch_first=True, padding_value=-100)
    bio_y = pad_sequence([torch.tensor(x["bio_y"]) for x in batch], batch_first=True, padding_value=-100)
    in_event_y = pad_sequence([torch.tensor(x["in_event_y"]) for x in batch], batch_first=True, padding_value=-100)

    tag_id = pad_sequence([torch.tensor(x["tag_id"]) for x in batch], batch_first=True, padding_value=0)
    parent_tag_id = pad_sequence([torch.tensor(x["parent_tag_id"]) for x in batch], batch_first=True, padding_value=0)

    num_feats = pad_sequence([torch.tensor(x["num_feats"]) for x in batch], batch_first=True, padding_value=0.0)
    bool_feats = pad_sequence([torch.tensor(x["bool_feats"]) for x in batch], batch_first=True, padding_value=0.0)

    # ---- 4) Node mask (valid nodes only) ----
    node_mask = field_y != -100

    return {
        "enc": enc,
        "node_offsets": node_offsets,
        "node_mask": node_mask,
        "field_y": field_y,
        "bio_y": bio_y,
        "in_event_y": in_event_y,
        "tag_id": tag_id,
        "parent_tag_id": parent_tag_id,
        "num_feats": num_feats,
        "bool_feats": bool_feats
    }
    
