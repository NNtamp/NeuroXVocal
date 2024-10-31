import argparse
import pandas as pd
import joblib
import os

"""
This script processes a CSV file for standardization using a pre-trained scaler.
The script performs the following steps:

1. Loads a CSV file specified by the user.
2. Excludes the 'patient_id' column and fills any missing values in the dataset with
   the mean of the corresponding column.
3. Loads a pre-trained StandardScaler model saved as a .pkl file to standardize
   the dataset based on the saved mean and scale values.
4. Adds the 'patient_id' column back to the standardized data.
5. Saves the processed CSV to an output path specified by the user.
"""

def process_csv(input_csv, scaler_path, output_csv):
    df = pd.read_csv(input_csv)
    patient_ids = df['patient_id']
    features = df.drop(columns=['patient_id'])
    features = features.apply(lambda x: x.fillna(x.mean()), axis=0)
    scaler = joblib.load(scaler_path)
    standardized_features = scaler.transform(features)
    standardized_df = pd.DataFrame(standardized_features, columns=features.columns)
    standardized_df['patient_id'] = patient_ids
    standardized_df.to_csv(output_csv, index=False)
    print(f"Processed CSV saved to '{output_csv}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file for standardization using a saved scaler.")
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("scaler_path", help="Path to the saved scaler (.pkl file).")
    parser.add_argument("output_csv", help="Path to save the processed CSV file.")
    args = parser.parse_args()
    process_csv(args.input_csv, args.scaler_path, args.output_csv)
