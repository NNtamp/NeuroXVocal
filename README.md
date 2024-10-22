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





