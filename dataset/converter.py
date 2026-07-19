import pandas as pd

def convert_tsv_to_csv(tsv_path, csv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    df.to_csv(csv_path, index=False)

# Example usage
convert_tsv_to_csv('seeds_dataset.tsv', 'seeds_dataset.csv')