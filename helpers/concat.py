import os
import csv
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" # @Yhilal02 These will need to change depending on where the data will be taken from
CLEANED = ROOT / "data" / "cleaned"

def get_headers(fp: str) -> list[str]:
    with open(fp, 'r', newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
            return headers
        except Exception as e:
            print(f"Failed to extract data columns from headers: {e}")
    return []


def fuse(raw_fp:str, target: str, files: list[str]) -> None:
    """
    target is the fp where you want to write out the result
    """
    dataframes = []
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(raw_fp, file)
            # Read the CSV file and append it to the list
            df = pd.read_csv(file_path)
            df["source"] = Path(file).stem
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(os.path.join(target, "full_data.csv"), index=False)


if __name__ == "__main__":

    files = [f.name for f in RAW.glob("*.csv")]
    
    #fetch headers from first csv (other files should have the same headers)... 
    headers = get_headers(os.path.join(RAW, files[0]))
    fuse(raw_fp=str(RAW), 
         target=str(CLEANED), 
         files=files)



