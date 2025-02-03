import pandas as pd
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
import soundfile as sf
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

# Loads data, reduces noise, extracts features, standardizes, and reduces the dimensionality of the data

def load_audio(filename):
    return sf.read(filename)

def reduce_noise(audio, sample_rate, output_file='cleaned.wav'):
    reduced_noise = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=.75)

    # Save the cleaned audio to a new file
    output_file = "../cleaned_output.wav"
    sf.write(output_file, reduced_noise, sample_rate)

    audio = AudioSegment.from_wav(output_file)
    return audio

# Read recorded keystrokes (labels)
# csv format: is pressed (1 for press, 0 for release), timestamp in ms, key
# For every press, search for the release of the same key, save the key, the start time, and release time - press time
def read_labels(file_name: str):
    label_file = open(file_name, 'r')

    i = 0 # oldest non-released key
    key_events = [] # list of labels and timestamps

    first_line = True

    for line in label_file:
        if first_line:
            first_line = False
            continue

        line = line.strip()

        is_press, timestamp, event_key = line.split('\t')

        event_key = event_key.lower()

        if is_press == '1':
            key_events.append({'key': event_key, 'start': int(timestamp), 'end': ''})
            continue
        if is_press == '0':
            for event in key_events[i:]:
                if event['key'] == event_key:
                    event['end'] = int(timestamp)
                    i += 1
                    break

    label_file.close()
    return key_events


def load_data(label_file, audio_file="audio.wav"):
    audio, sample_rate = load_audio(audio_file)
    cleaned_audio = reduce_noise(audio, sample_rate)
    
    labels = None
    if label_file:
        labels = read_labels(label_file)

    return cleaned_audio, labels, sample_rate

# Segment audio based on labels
# padding: amount of time before press and after release to include
def labeled_audio_segmentation(labels, audio, padding=20):
    keystrokes = [] # list of segments of audio
    labels_list = [] # list of dicts that contain key pressed, time pressed, time released
    for event in labels:

        label = event['key']
        start = event['start'] - padding

        # If there is no release, then the event will be skipped
        try:
            end = event['end'] + padding
        except TypeError:
            continue

        key_audio = audio[start:end]
        keystrokes.append(key_audio)
        labels_list.append(label)

    return keystrokes, labels_list


"""
Written by Shoyo Inokuchi (June 2019)

Scripts for the acoustic keylogger surrounding feature extraction.
Repository is located at: https://github.com/shoyo-inokuchi/acoustic-keylogger
"""
from librosa.feature import mfcc


def extract_features(keystroke, sr=44100, n_mfcc=16, n_fft=441, hop_len=110):
    """Return an MFCC-based feature vector for a given keystroke."""
    spec = mfcc(y=keystroke.astype(float),
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft, # n_fft=220 for a 10ms window
                hop_length=hop_len, # hop_length=110 for ~2.5ms
                )
    return spec.flatten()


def convert_to_array(list_of_keys):
    list_of_arrays = []
    for key in list_of_keys:
        key_array = np.array(key.get_array_of_samples())
        list_of_arrays.append(extract_features(key_array, sr=key.frame_rate))
    
    return list_of_arrays

def scale_features(data: pd.DataFrame):
    sc = StandardScaler()
    return sc.fit_transform(data), sc

def dim_reduction(data: pd.DataFrame):
    pca = PCA(n_components=20)
    return pca.fit_transform(data), pca

def one_hot(labels):
    labels_reshaped = np.array(labels).reshape(-1, 1)

    encoder = OneHotEncoder()
    encoded_labels = encoder.fit_transform(labels_reshaped)
    encoded_labels = encoded_labels.toarray()
    return encoded_labels, encoder

def preprocess_data(data_dir='data', labeled=True):
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
        label_file = data_dir + '/' + base + '.tsv'
        audio_file = data_dir + '/' + base + '.wav'

        audio, labels, _ = load_data(label_file, audio_file)
        segmented_audio, seg_labels = labeled_audio_segmentation(labels, audio)

        key_df = pd.DataFrame(convert_to_array(segmented_audio))
        key_df.fillna(0, inplace=True)
        key_df['label'] = seg_labels

        dataframe = pd.concat([dataframe, key_df], ignore_index=True)
    
    features = dataframe.drop('label', axis=1)
    labels = dataframe['label']

    scaled_features, scaler = scale_features(features)
    reduced_features, pca = dim_reduction(scaled_features)
    one_hot_labels, ohe = one_hot(labels)

    label_list = (labels, one_hot_labels)

    processed_df = pd.DataFrame(reduced_features)

    return processed_df, label_list, {'scaler': scaler, 'pca': pca, 'encoder': ohe}


def main():
    print(read_labels('data/data1.tsv'))

if __name__ == '__main__':
    main()