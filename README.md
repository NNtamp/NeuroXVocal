
## 🧠⚙️[A] Classifier Training

First of all install all the necessary requirements:
```
pip install -r requirements.txt
```
For the training of the classifier data from the [IS2021 ADReSSo Challenge](https://edin.ac/3p1cyaI) was used. Initial form of data was audio files of .wav format.

The data should be organized in the following structure for the succesfull training following Steps I and II of data preparation.

```
train
    ├── ad
    │   ├── {patient_id1}.txt
    │   └── {patient_id2}.txt
    │   └── {patient_id...}.txt
    │   └── audio_embeddings_ad.csv 
    │   └── audio_features_ad.csv            
    ├── cn
    │   ├── {patient_id1}.txt
    │   └── {patient_id2}.txt
    │   └── {patient_id...}.txt
    │   └── audio_embeddings_cn.csv 
    │   └── audio_features_cn.csv 
```
### I. Data extraction

1. Extract text from audio:
```
python src/data_extraction/transcribe_audio.py --path/to/audio/files
```
2. Extract audio features:
```
python src/data_extraction/extract_audio_features.py --path/to/audio/files --output_csv --path/to/create/the/audio_features.csv  
```
3. Extract audio embeddings:
```
python src/data_extraction\extract_audio_embeddings.py --path/to/audio/files --path/to/create/the/audio_embeddings.csv  
```
### II. Data process

1. Text processing:
```
python src/data_processing/preprocess_texts.py --path/to/txt/files --path/to/save/processed/txt/files
```
2. Audio features processing using scaler params:
```
python src/data_processing/preprocess_audio_features.py --path/to/audio_features.csv --path/to/save/processed/audio_features.csv --scaler_path src/inference/scaler_params_audio_features.pkl 
```
3. Audio embeddings processing using scaler params:
```
python src/data_processing/preprocess_audio_emb.py --path/to/audio_embeddings.csv --scaler_path src/inference/scaler_params_audio_emb.pkl --path/to/save/processed/audio_embeddings.csv
```
### III. NeuroXVocal Training

After setting up the values in src/train/config.py run:

```
python src/train/main.py  
```





