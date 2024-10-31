import os
import sys
import argparse
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
import pandas as pd
from tqdm import tqdm

'''
Script Description:

This script processes audio files in a specified directory to extract embeddings using the Wav2Vec 2.0 model. 
Each audio file (.wav format) undergoes preprocessing, including mono conversion and resampling to 16kHz, 
before embeddings are extracted using a pretrained Wav2Vec 2.0 model. The resulting embeddings for each 
file are saved to a CSV file, with each row containing a unique patient ID and the corresponding embedding values.

Command-Line Arguments:

- data_path: Path to the directory containing .wav files for processing.
- output_csv: (Optional) Specifies the name of the output CSV file for saving embeddings. Default is 'audio_embeddings.csv'.
  
'''


def extract_embeddings(audio_path, model, processor, device):
    speech_array, sampling_rate = torchaudio.load(audio_path)
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0)
        print(f"Converted stereo to mono for {audio_path}")
    else:
        speech_array = speech_array.squeeze(0)

    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)
        sampling_rate = 16000

    speech = speech_array.numpy() 

    inputs = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state
    embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()

    return embedding

def process_audio_files(data_path, output_csv, model_name='facebook/wav2vec2-base-960h'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeddings = []
    for root, dirs, files in os.walk(data_path):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.lower().endswith('.wav'):
                audio_path = os.path.join(root, file)
                patient_id = os.path.splitext(file)[0]
                try:
                    embedding = extract_embeddings(audio_path, model, processor, device)
                    all_embeddings.append({'patient_id': patient_id, 'embedding': embedding})
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    continue  # Skip to the next file

    if len(all_embeddings) == 0:
        print("No embeddings were extracted. Please check for errors.")
        return

    df = pd.DataFrame(all_embeddings)
    # Expand the 'embedding' column into separate columns
    embedding_cols = pd.DataFrame(df['embedding'].tolist())
    embedding_cols['patient_id'] = df['patient_id']
    embedding_cols.to_csv(output_csv, index=False)
    print(f"Embeddings saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Extract Wav2Vec 2.0 embeddings from audio files.")
    parser.add_argument('data_path', help='Path to the directory containing .wav files.')
    parser.add_argument('--output_csv', default='audio_embeddings.csv', help='Output CSV file name.')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: The directory {args.data_path} does not exist.")
        sys.exit(1)

    process_audio_files(
        data_path=args.data_path,
        output_csv=args.output_csv
    )

if __name__ == '__main__':
    main()
