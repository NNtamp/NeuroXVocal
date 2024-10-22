import os
import sys
import argparse
import whisper

'''
Command-Line Arguments:

- data_path: Path to the directory containing .wav files.
'''

def transcribe_audio_files(data_path):
    model = whisper.load_model("base")

    output_base = os.path.join(os.path.dirname(data_path), 'extracted_data')

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, data_path)
                output_dir = os.path.join(output_base, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.txt')
                print(f"Transcribing {audio_path}...")
                result = model.transcribe(audio_path)
                text = result["text"]
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Saved transcription to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe .wav files to text using Whisper.")
    parser.add_argument('data_path', help='Path to the directory containing .wav files.')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: The directory {args.data_path} does not exist.")
        sys.exit(1)

    transcribe_audio_files(args.data_path)

if __name__ == '__main__':
    main()
