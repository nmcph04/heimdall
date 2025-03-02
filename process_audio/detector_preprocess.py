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

# Encodes an array with binary labels with one-hot encoding
def encode_labels(labels: np.ndarray):
    encoded = []
    for label in labels:
        encoded.append([0, 1] if label == 1 else [1, 0])
    
    return np.array(encoded)

# offset: will also use +-x seconds from event_start as positive samples 
#   must be less than segment length (20ms by default)
#   should be less than or equal to half of segment length
def detector_audio_segmentation(labels, audio, sr=44100, offset=0.01):
    samples = []
    labels_list = []
    sample_length = int(SEGMENT_LEN * sr)
    offset_ms = int(offset * 1000.)
    audio_len = len(audio)

    if (offset / 1000.) >= sample_length:
        raise Exception(f"Offset must be less than segment length({SEGMENT_LEN}), but got {offset}!")

    positive_times = []

    # Get positive samples
    for event_start in labels:
        event_start = int((event_start / 1000.) * sr) # Change event_start from ms to samples

        # from -offset to offset ms
        for start_offset in range(offset_ms * -1, offset_ms + 1):
            start_offset /= 1000.
            offset_samples = int(start_offset * sr)

            start = event_start + offset_samples
            if event_start < 0:
                continue

            # limits end of sample to end of audio
            end = min(start + sample_length, audio_len)

            samples.append(audio[start:end])
            labels_list.append(1)

        # add offset range to list so they can be skipped for negative samples
        offset_range_start = int(event_start - (offset_ms * -1 / 1000.))
        offset_range_end = int(event_start + (offset_ms / 1000.))
        positive_times.append((offset_range_start, offset_range_end))
    
    # Get negative samples
    for idx in range(0, audio_len, sample_length):
        invalid = False
        end = min(idx + sample_length, audio_len)

        # If selected segment is in a positive sample, it will be skipped
        for segment in positive_times:
            # Will stop the positive sample checking if the current range ends before it starts
            if end <= segment[0]:
                break
            elif (segment[0] <= idx <= segment[1]) and (segment[0] <= end <= segment[1]):
                invalid = True
                break

        if not invalid:
            sample = audio[idx:end]

            # Pads the end with zeros if it is too small (e.g. it was at the end of the recording)
            if len(sample) != sample_length:
                pad_amt = sample_length - len(sample)
                sample = np.pad(sample, (0, pad_amt), 'constant', constant_values=(0))
            samples.append(sample)
            labels_list.append(0)
                

    return samples, labels_list

# Preprocesses data for the detector model
# Only processes labeled data
def preprocess_for_detector(data_dir='data/', init_transformers=True):
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

    if init_transformers:
        encoded_labels = encode_labels(labels)

        scaled_features, scaler = scale_features(features)
        reduced_features, pca = dim_reduction(scaled_features)

        return reduced_features, encoded_labels, {'scaler': scaler, 'pca': pca}

def main():
    df = preprocess_for_detector()
    print(df.tail())

if __name__ == '__main__':
    main()