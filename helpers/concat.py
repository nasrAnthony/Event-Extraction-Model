import os
import csv
import pandas as pd



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
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(os.path.join(target, "full_data.csv"), index=False)


if __name__ == "__main__":

    raw_data_folder = os.path.join(os.getcwd(), "../data", "raw")
    cleaned_data_folder = os.path.join(os.getcwd(), "../data", "cleaned")
    files = os.listdir(raw_data_folder)
    
    #fetch headers from first csv (other files should have the same headers)... 
    headers = get_headers(os.path.join(raw_data_folder, files[0]))
    fuse(raw_fp=raw_data_folder, 
         target=cleaned_data_folder, 
         files=files)



