import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEANED = ROOT / "data" / "cleaned"

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mostly added checks
    Input: pd.Dataframe
    Output: cleaned pd.Dataframe
    """
    df = df.drop_duplicates() # removing exact duplicates
    
    # extra whitespace removal
    df.columns = df.columns.str.strip()
    df['tag'] = df['tag'].str.strip()
    df['text_context'] = df['text_context'].str.strip()
    df['parent_tag'] = df['parent_tag'].str.strip()
    df['label'] = df['label'].str.strip()
    
    df["link"] = df["link"].astype("string")
    
    return df

if __name__ == "__main__":
    
    # folder creation if needed
    try:
        CLEANED.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating folder '{CLEANED}': {e}")
    
    # cleaning/saving loop for csv files
    for csv_path in RAW.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df_clean = clean(df)
        
        df_clean.to_csv(CLEANED/csv_path.name, index=False)