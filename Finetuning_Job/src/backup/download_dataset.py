
from datasets import load_dataset
import pandas as pd
import os

def download_sample():
    print("Downloading dataset sample...")
    # Download the full dataset (approx 26k rows)
    ds = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset', split='train')
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Bitext-customer-support-llm-chatbot-training-dataset.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = ds.to_pandas()
    df.to_csv(output_path, index=False)
    print(f"Saved sample to {output_path}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nColumns:", df.columns.tolist())

if __name__ == "__main__":
    download_sample()
