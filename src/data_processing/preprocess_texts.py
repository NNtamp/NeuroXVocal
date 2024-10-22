import os
import sys
import argparse
import re

"""
Arguments:
    input_dir (str):
        - Description: The directory containing the input .txt files to be processed.
        - Requirement: Must exist and contain .txt files.
    output_dir (str):
        - Description: The directory where the processed .txt files will be saved.
        - Requirement: Will be created if it does not exist.

"""

def preprocess_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)

    return text

def process_text_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.txt'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            processed_text = preprocess_text(text)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(processed_text)

            print(f"Processed {filename}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess text files in a directory.')
    parser.add_argument('input_dir', help='Path to the input directory containing .txt files.')
    parser.add_argument('output_dir', help='Path to the output directory to save processed .txt files.')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        sys.exit(1)
    process_text_files(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()
