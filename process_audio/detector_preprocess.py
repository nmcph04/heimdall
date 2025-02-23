import pandas as pd
import numpy as np
import os
from preprocess_data import load_audio, reduce_noise, convert_to_array, scale_features, dim_reduction, one_hot

# 20ms length for each feature
SEGMENT_LEN = 0.02

def read_labels(file_name: str):
    label_file = open(file_name, 'r')

    positive_events = []
    first_line = True
    for line in label_file:
        if first_line:
            first_line = False
            continue

        line = line.strip()

        is_press, timestamp_str, _ = line.split('\t')
        timestamp = int(timestamp_str)

        if is_press == '1':
            positive_events.append(timestamp)
    
    return positive_events

def load_data(label_file="labels.tsv", audio_file="audio.wav"):
    audio, sample_rate = load_audio(audio_file)
    cleaned_audio = reduce_noise(audio, sample_rate)
    
    key_events = None
    if label_file:
        key_events = read_labels(label_file)

    return cleaned_audio, key_events, sample_rate

def detector_audio_segmentation(labels, audio, sr=44100, ms_pad=5):
    samples = []
    labels_list = []
    padding = int(sr * (ms_pad / 1000.))
    sample_length = int(SEGMENT_LEN * sr)
    audio_len = len(audio)

    positive_times = []

    # Get positive samples
    for event_start in labels:
        event_start = int((event_start / 1000.) * sr) # Change event_start from ms to samples
        # Doesn't let the start timestamp be negative
        padded_start = max(0, event_start - padding)

        end = min(padded_start + sample_length, audio_len)

        samples.append(audio[padded_start:end])
        labels_list.append(1)
        positive_times.append((padded_start, end))
    
    # Get negative samples
    for idx in range(0, audio_len, sample_length):
        invalid = False
        start = max(0, idx - padding)
        end = min(start + sample_length, audio_len)

        # If selected segment is in a positive sample, it will be skipped
        for segment in positive_times:
            # Will stop the positive sample checking if the current range ends before it starts
            if end <= segment[0]:
                break
            elif (segment[0] <= start <= segment[1]) and (segment[0] <= end <= segment[1]):
                invalid = True
                break
        if not invalid:
            sample = audio[start:end]
            # Pads the end with zeros if it is too small (e.g. it was at the end of the recording)
            if len(sample) != sample_length:
                pad_amt = sample_length - len(sample)
                sample = np.pad(sample, (0, pad_amt), 'constant', constant_values=(0))
            samples.append(sample)
            labels_list.append(0)
                

    return samples, labels_list

# Preprocesses data for the detector model
# Only processes labeled data
def preprocess_for_detector(data_dir='data/'):
    base_names = []
    extensions = ['.txt', '.wav']
    filenames = os.listdir(data_dir)

    for file in filenames:
        base, ext = os.path.splitext(file)
        # Appends file base name to base_names if it has one of the two extensions and is not already in base_names
        if ext in extensions and base not in base_names:
            base_names.append(base)

    # loads the data from every pair of txt and wav files in the data_dir    
    dataframe = pd.DataFrame()

    for base in base_names:
        print(f'Processing {base} file(s)...', end='', flush=True)
        label_file = data_dir + '/' + base + '.tsv'
        audio_file = data_dir + '/' + base + '.wav'

        audio, labels, sr = load_data(label_file, audio_file)
        segmented_audio, seg_labels = detector_audio_segmentation(labels, audio, sr)

        key_df = pd.DataFrame(convert_to_array(segmented_audio, n_fft=64))
        key_df['label'] = seg_labels

        dataframe = pd.concat([dataframe, key_df], ignore_index=True)

        print(" Finished.")

    dataframe.fillna(0, inplace=True)
    features = dataframe.drop('label', axis=1)
    labels = dataframe['label']

    encoder = one_hot(labels)

    scaled_features, scaler = scale_features(features)
    reduced_features, pca = dim_reduction(scaled_features)
    processed_df = pd.DataFrame(reduced_features)

    return processed_df, labels, {'scaler': scaler, 'pca': pca, 'encoder': encoder}

def main():
    df = preprocess_for_detector()
    print(df.tail())

if __name__ == '__main__':
    main()