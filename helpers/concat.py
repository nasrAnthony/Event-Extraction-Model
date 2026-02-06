import os
import csv
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / "data" / "cleaned"
CONCAT = ROOT / "data" 

def get_headers(fp: str) -> list[str]:
    with open(fp, 'r', newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
            return headers
        except Exception as e:
            print(f"Failed to extract data columns from headers: {e}")
    return []


def fuse(clean_fp:str, target: str, files: list[str]) -> None:
    """
    target is the fp where you want to write out the result
    """
    dataframes = []
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(clean_fp, file)
            # Read the CSV file and append it to the list
            df = pd.read_csv(file_path)
            df["source"] = Path(file).stem
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(os.path.join(target, "full_data.csv"), index=False)


if __name__ == "__main__":

    files = [f.name for f in CLEANED.glob("*.csv")]
    
    #fetch headers from first csv (other files should have the same headers)... 
    headers = get_headers(os.path.join(CLEANED, files[0]))
    fuse(clean_fp=str(CLEANED), 
         target=str(CONCAT), 
         files=files)



