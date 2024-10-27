# This directory contains the training data for the [IS2021 ADReSSo Challenge](https://edin.ac/3p1cyaI).

The data are in the following directories structure and files:

```
diagnosis
└── train
    ├── audio
    │   ├── ad
    │   └── cn
    └── segmentation
        ├── ad
        └── cn
```

They contain the enhanced, volume normalised audio data for the
diagnosis and MMSE score prediction tasks, and a table of MMSE scores
for model training (adresso-train-mmse-scores.csv). The abbreviation
'cn' denotes 'control' patients, and 'ad' patients with a (probable)
Alzheimer's dementia diagnosis.

Also included are the utterance segmentation files (diarisation), in
CSV format. These files are for those who choose to do the segmented
prediction sub-task. The segmented prediction and speech-only
sub-tasks will be assessed separately.

Audio to text:
python src/data_extraction/transcribe_audio.py data/ADReSSo21_audio/diagnosis/train/audio

Audio features:
python src/data_extraction/extract_audio_features.py data/ADReSSo21_audio/diagnosis/train/audio/cn --output_csv data/ADReSSo21_audio/diagnosis/train/extracted_data/cn/audio_features.csv  

Process text:
python src/data_processing/preprocess_texts.py data/ADReSSo21_audio/diagnosis/train/extracted_data/cn data/ADReSSo21_audio/diagnosis/train/processed_data/cn

Process audio features:
python preprocess.py --input_path <input_csv_path> --output_path <output_directory> --scaler_path <scaler_pkl_path>
python src/data_processing/preprocess_audio_features.py --input_path data/ADReSSo21_audio/diagnosis/train/extracted_data/cn/audio_features.csv --output_path data/ADReSSo21_audio/diagnosis/train/processed_data/cn --scaler_path src/inference/scaler_params.pkl 

Training:
python src/train/main.py  

Inf Adr:
python inf_adr.py --text_data_dir "C:\Users\30697\Desktop\Personal Projects\NeuroXVocal\data\ADReSSo21_audio\diagnosis\test-dist\processed_data" --audio_csv_path "C:\Users\30697\Desktop\Personal Projects\NeuroXVocal\data\ADReSSo21_audio\diagnosis\test-dist\processed_data\audio_features.csv" --model_path "C:\Users\30697\Desktop\Personal Projects\NeuroXVocal\results\results\model_fold1_epoch48.pth" --output_dir "C:\Users\30697\Desktop\Personal Projects\NeuroXVocal\data\ADReSSo21_audio\diagnosis\test-dist\submissions"

Rearrange orders to match template:
python rearrange_predictions.py --template_path "C:\Users\30697\Desktop\Personal Projects\NeuroXVocal\data\ADReSSo21_audio\diagnosis\test-dist\test_results_task1.csv" --predictions_path "C:\Users\30697\Desktop\Personal Projects\NeuroXVocal\data\ADReSSo21_audio\diagnosis\test-dist\submissions\test_results-task1-1_48.csv"






