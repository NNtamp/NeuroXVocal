import pandas as pd
import numpy as np
import argparse
import os
import joblib

"""
Preprocessing Script for CSV Files

This script performs the following preprocessing steps:
1. Drops unnecessary columns from the dataset: 
    ['jitter_local', 'shimmer_local', 'formant_1_mean', 'formant_1_std', 
     'formant_2_mean', 'formant_2_std', 'formant_3_mean', 'formant_3_std', 'class']
2. Temporarily drops the 'patient_id' column for processing.
3. Checks for missing values in each column and fills any missing values with the average of the remaining values.
4. Standardizes the values in the dataset based on a pre-trained scaler loaded from a pickle file.
5. Adds back the 'patient_id' column after standardization.
6. Saves the processed dataset with the same name to a specified output directory.

Arguments:
- input_path: Path to the input CSV file.
- output_path: Directory path where the processed file will be saved (same name as input).
- scaler_path: Path to the pickle file (scaler_params.pkl) for standardization.
"""

def preprocess_csv(input_path, output_path, scaler_path):
    df = pd.read_csv(input_path)
    df = df.drop(['jitter_local', 'shimmer_local', 'formant_1_mean', 
                  'formant_1_std', 'formant_2_mean', 'formant_2_std', 
                  'formant_3_mean', 'formant_3_std', 'class'], axis=1)
    
    patient_ids = df['patient_id']
    df = df.drop(['patient_id'], axis=1)
    df = df.apply(lambda x: x.fillna(x.mean()) if x.isnull().any() else x)
    scaler = joblib.load(scaler_path)
    scaled_features = scaler.transform(df)
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns)
    df_scaled['patient_id'] = patient_ids
    output_file = os.path.join(output_path, os.path.basename(input_path))
    df_scaled.to_csv(output_file, index=False)
    print(f"Processed file saved at: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CSV files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the processed CSV file.")
    parser.add_argument("--scaler_path", type=str, required=True, help="Path to the scaler_params_audio_fetures.pkl file.")
    args = parser.parse_args()
    
    preprocess_csv(args.input_path, args.output_path, args.scaler_path)
