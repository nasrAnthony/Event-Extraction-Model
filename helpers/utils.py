import yaml
import numpy as np
from pathlib import Path


def load_config(path=None):
    """
    load config.yaml from the repo root
    pass a custom path if needed, otherwise resolve automatically
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
    
def compute_num_stats(train_df, cols):
    """
    compute mean and std of numeric columns from training data
    returns float32 arrays for mean and std
    """
    x = train_df[cols].fillna(0).values.astype("float32")
    
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype("float32")
    return mean.astype("float32"), std.astype("float32")