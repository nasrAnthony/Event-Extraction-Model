import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# Label Merging -------------------------------------------

# map of labels to be merged
LABEL_MERGE_MAP = {
    # keep
    "Other": "Other",

    # Name variants
    "Name": "Name", "NameLink": "Name", "NameLocation": "Name",

    # Date variants
    "Date": "Date", "DateTime": "Date",

    # Time variants
    "Time": "Time", "StartTime": "Time", "EndTime": "Time",
    "StartEndTime": "Time", "TimeLocation": "Time",
    
    # Location variants
    "Location": "Location",
    
    # Price variants
    "Price": "Price",
    
    # Description variants
    "Description": "Description", "Desc": "Description", "Details": "Description",
}

def merge_labels(df, label_col="label", mapping=LABEL_MERGE_MAP, default_to_other=True):
    df = df.copy()
    mapping = mapping or {}

    def _map(x):
        x = str(x)
        if x in mapping:
            return mapping[x]
        return "Other" if default_to_other else x

    df[label_col] = df[label_col].map(_map)
    return df


# Dataset Class ------------------------------------------

class PageDataset(Dataset):
    """
    groups rows/nodes by source and tokenizes text 
    applies numeric normalization using provided mean and std
    each dataset item is a whole webpage
    """
    def __init__(self, df, tokenizer, tag_vocab, parent_tag_vocab, 
                 num_cols, bool_cols, mean=None, std=None, max_tokens=64):
        self.pages = []

        # mean and std (empty for safety/normal distribution)
        if mean is None:
            mean = np.zeros((len(num_cols),), dtype="float32")
        if std is None:
            std = np.ones((len(num_cols),), dtype="float32")
        self.num_mean = mean
        self.num_std = std

        for src, g in df.groupby("source", sort=False):
            g = g.sort_values("rendering_order").reset_index(drop=True) # extra sort

            # TEXT FEATURES: tokenize
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
            num = (num - mean) / std

            # overall page dict
            self.pages.append({
                "input_ids": enc["input_ids"],  # transformer output
                "attention_mask": enc["attention_mask"],

                "bio_y": g["bio"].astype(int).tolist(),  # boundaries

                "tag_id": [tag_vocab[str(x)] for x in g["tag"]],
                "parent_tag_id": [parent_tag_vocab[str(x)] for x in g["parent_tag"]],

                "num_feats": num,
                "bool_feats": g[bool_cols].astype(int).values.astype("float32")
            })

    def __len__(self):
        return len(self.pages) # number of pages

    def __getitem__(self, idx):
        return self.pages[idx] # returns a single page
    

# More Data Manipulation -----------------------------------------------

def combine_pages(batch, tokenizer):
    """
    function deals with flattening text and padding nodes
    batch: list of pages
    returns a padded tensor with all node/page data
    """
    B = len(batch)
    node_offsets = []

    # flattening all text data across all pages [total nodes, token_length]
    flat_text = [
        {"input_ids": ids, "attention_mask": mask}
        for x in batch
        for ids, mask in zip(x["input_ids"], x["attention_mask"])
    ]
    enc = tokenizer.pad(flat_text, padding =True, return_tensors="pt") # padding tokens to match max length

    # track where each pager's nodes start/end in a flat list
    lengths = [len(x["input_ids"]) for x in batch]  # num of nodes per page
    offsets = torch.cumsum(torch.tensor([0] + lengths), dim=0)  # to indicate where pages start
    node_offsets = [(offsets[i].item(), offsets[i+1].item()) for i in range(B)]  # page ranges

    # applying pdding
    bio_y = pad_sequence([torch.tensor(x["bio_y"]) for x in batch], batch_first=True, padding_value=-100)

    tag_id = pad_sequence([torch.tensor(x["tag_id"]) for x in batch], batch_first=True, padding_value=0)
    parent_tag_id = pad_sequence([torch.tensor(x["parent_tag_id"]) for x in batch], batch_first=True, padding_value=0)
    num_feats = pad_sequence([torch.tensor(x["num_feats"]) for x in batch], batch_first=True, padding_value=0.0)
    bool_feats = pad_sequence([torch.tensor(x["bool_feats"]) for x in batch], batch_first=True, padding_value=0.0)

    # true for real nodes, false for padding
    node_mask = bio_y != -100

    return {
        "enc": enc,
        "node_offsets": node_offsets,
        "node_mask": node_mask,
        "bio_y": bio_y,
        "tag_id": tag_id,
        "parent_tag_id": parent_tag_id,
        "num_feats": num_feats,
        "bool_feats": bool_feats
    }
    
