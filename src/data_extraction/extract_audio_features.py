import os
import sys
import argparse
import librosa
import numpy as np
import pandas as pd
import parselmouth
from scipy.signal import find_peaks

'''
Descriptions of Features:

- duration: Total length of the audio recording in seconds.
- total_speech_time: Total time the patient is speaking (excluding pauses).
- speech_pause_ratio: Ratio of speaking time to total recording duration.
- num_pauses: Number of pauses detected in the speech.
- total_pause_duration: Cumulative duration of all pauses.
- avg_pause_duration: Average duration of pauses.
- max_pause_duration: Duration of the longest pause.
- pause_duration_std: Standard deviation of pause durations.
- pitch_mean: Mean of the fundamental frequency (pitch).
- pitch_std: Standard deviation of the fundamental frequency.
- pitch_range: Range (max - min) of the fundamental frequency.
- intensity_mean: Mean of speech intensity (loudness).
- intensity_std: Standard deviation of speech intensity.
- intensity_range: Range of speech intensity.
- articulation_rate: Rate of speech excluding pauses.
- speaking_rate: Estimated syllables per second during speaking time.
- spectral_centroid_mean: Mean spectral centroid (center of mass of the spectrum).
- spectral_centroid_std: Standard deviation of spectral centroid.
- zcr_mean: Mean zero-crossing rate (measure of signal noisiness).
- zcr_std: Standard deviation of zero-crossing rate.
- jitter_local: Local jitter value (variability in pitch frequency).
- shimmer_local: Local shimmer value (variability in amplitude).
- hnr_mean: Mean harmonics-to-noise ratio (measure of voice hoarseness).
- formant_1_mean: Mean of the first formant frequency.
- formant_1_std: Standard deviation of the first formant frequency.
- formant_2_mean: Mean of the second formant frequency.
- formant_2_std: Standard deviation of the second formant frequency.
- formant_3_mean: Mean of the third formant frequency.
- formant_3_std: Standard deviation of the third formant frequency.
- mfcc_1_mean to mfcc_13_mean: Mean values of the first 13 MFCCs.
- mfcc_1_std to mfcc_13_std: Standard deviation of the first 13 MFCCs.

Command-Line Arguments:

- data_path: Path to the directory containing .wav files.
- --output_csv: Output CSV file name where features will be saved. Default is 'audio_features.csv'.
- --sample_rate: Sample rate for audio processing in Hz. Default is 22050.
- --frame_length: Frame length for audio analysis in number of samples. Default is 2048.
- --hop_length: Hop length for audio analysis in number of samples. Default is 512.
- --silence_threshold: Silence threshold as a fraction (e.g., 0.1 for 10th percentile). Used for silence detection. Default is 0.1.
'''

def extract_features(audio_path, sr=22050, frame_length=2048, hop_length=512, silence_threshold=0.1):
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])
    threshold = np.percentile(energy, silence_threshold * 100) 
    silence_frames = energy < threshold
    speech_frames = ~silence_frames
    total_speech_frames = np.sum(speech_frames)
    total_speech_time = total_speech_frames * hop_length / sr
    silence_durations = []
    current_silence = 0
    for is_silence in silence_frames:
        if is_silence:
            current_silence += 1
        elif current_silence > 0:
            silence_durations.append(current_silence * hop_length / sr)
            current_silence = 0
    num_pauses = len(silence_durations)
    total_pause_duration = sum(silence_durations)
    avg_pause_duration = np.mean(silence_durations) if silence_durations else 0
    max_pause_duration = max(silence_durations) if silence_durations else 0
    pause_duration_std = np.std(silence_durations) if silence_durations else 0
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    pitches = pitches.flatten()
    magnitudes = magnitudes.flatten()
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitches) if len(pitches) > 0 else 0
    pitch_std = np.std(pitches) if len(pitches) > 0 else 0
    pitch_range = np.ptp(pitches) if len(pitches) > 0 else 0 
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    intensity = librosa.amplitude_to_db(S, ref=np.max)
    intensity_mean = np.mean(intensity)
    intensity_std = np.std(intensity)
    intensity_range = np.ptp(intensity)

    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    spectral_centroid_mean = np.mean(spectral_centroids)
    spectral_centroid_std = np.std(spectral_centroids)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    articulation_rate = total_speech_time / (duration - total_pause_duration) if (duration - total_pause_duration) > 0 else 0

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = find_peaks(onset_env, height=np.mean(onset_env))[0]
    estimated_syllables = len(peaks)
    speaking_rate = estimated_syllables / total_speech_time if total_speech_time > 0 else 0

    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch(time_step=hop_length/sr)
    point_process = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")
    local_jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 1.0 / 75, 1.0 / 500, 1.3)
    local_shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 1.0 / 75, 1.0 / 500, 1.3, 1.6)
    harmonicity = snd.to_harmonicity_cc(time_step=hop_length/sr)
    hnr_values = harmonicity.values
    hnr_mean = np.mean(hnr_values[hnr_values != 0]) if np.any(hnr_values != 0) else 0
    
    formant = snd.to_formant_burg(time_step=hop_length/sr)
    f1 = []
    f2 = []
    f3 = []
    for t in np.arange(0, snd.duration, hop_length / sr):
        try:
            f1.append(formant.get_value_at_time(1, t))
            f2.append(formant.get_value_at_time(2, t))
            f3.append(formant.get_value_at_time(3, t))
        except:
            continue
    f1 = np.array(f1)
    f2 = np.array(f2)
    f3 = np.array(f3)
    formant_1_mean = np.mean(f1) if len(f1) > 0 else 0
    formant_1_std = np.std(f1) if len(f1) > 0 else 0
    formant_2_mean = np.mean(f2) if len(f2) > 0 else 0
    formant_2_std = np.std(f2) if len(f2) > 0 else 0
    formant_3_mean = np.mean(f3) if len(f3) > 0 else 0
    formant_3_std = np.std(f3) if len(f3) > 0 else 0

    features = {
        'duration': duration,
        'total_speech_time': total_speech_time,
        'speech_pause_ratio': total_speech_time / duration if duration > 0 else 0,
        'num_pauses': num_pauses,
        'total_pause_duration': total_pause_duration,
        'avg_pause_duration': avg_pause_duration,
        'max_pause_duration': max_pause_duration,
        'pause_duration_std': pause_duration_std,
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'pitch_range': pitch_range,
        'intensity_mean': intensity_mean,
        'intensity_std': intensity_std,
        'intensity_range': intensity_range,
        'articulation_rate': articulation_rate,
        'speaking_rate': speaking_rate,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_centroid_std': spectral_centroid_std,
        'zcr_mean': zcr_mean,
        'zcr_std': zcr_std,
        'jitter_local': local_jitter,
        'shimmer_local': local_shimmer,
        'hnr_mean': hnr_mean,
        'formant_1_mean': formant_1_mean,
        'formant_1_std': formant_1_std,
        'formant_2_mean': formant_2_mean,
        'formant_2_std': formant_2_std,
        'formant_3_mean': formant_3_mean,
        'formant_3_std': formant_3_std,
    }

    for i, (mean, std) in enumerate(zip(mfccs_mean, mfccs_std)):
        features[f'mfcc_{i+1}_mean'] = mean
        features[f'mfcc_{i+1}_std'] = std

    return features

def process_audio_files(data_path, output_csv, sr, frame_length, hop_length, silence_threshold):
    all_features = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.wav'):

                audio_path = os.path.join(root, file)
                patient_id = os.path.splitext(file)[0]
                print(f"Processing {audio_path}...")
                try:
                    features = extract_features(
                        audio_path,
                        sr=sr,
                        frame_length=frame_length,
                        hop_length=hop_length,
                        silence_threshold=silence_threshold
                    )
                    features['patient_id'] = patient_id
                    features['class'] = os.path.basename(root) 
                    all_features.append(features)
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Extract audio features for dementia detection.")
    parser.add_argument('data_path', help='Path to the directory containing .wav files.')
    parser.add_argument('--output_csv', default='audio_features.csv', help='Output CSV file name.')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate for audio processing.')
    parser.add_argument('--frame_length', type=int, default=2048, help='Frame length for audio analysis.')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for audio analysis.')
    parser.add_argument('--silence_threshold', type=float, default=0.1, help='Silence threshold as a fraction (e.g., 0.1 for 10th percentile).')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: The directory {args.data_path} does not exist.")
        sys.exit(1)

    process_audio_files(
        data_path=args.data_path,
        output_csv=args.output_csv,
        sr=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        silence_threshold=args.silence_threshold
    )

if __name__ == '__main__':
    main()
