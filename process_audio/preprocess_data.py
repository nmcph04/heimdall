import pandas as pd
import numpy as np
import noisereduce as nr
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import librosa
import torch
from deep_learning_functions import load_model, load_transformers, transform_data, decode_binary_label

# Loads data, reduces noise, extracts features, standardizes, and reduces the dimensionality of the data

SEGMENT_LEN = 0.2 # each segment should be 200ms long

def load_audio(filename):
    return librosa.load(filename, sr=None)

def reduce_noise(audio, sample_rate, output_file='cleaned.wav'):
    reduced_noise = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=.75)
    return reduced_noise

# Convert shifted characters (!, @, etc.) to unshifted versions (1, 2, etc)
def map_shifted_keys(key):
    """
    Maps a shifted key (e.g., '~', '@') to its unshifted counterpart (e.g., ',', '1').
    
    :param key: A string representing the shifted key.
    :return: The corresponding unshifted character if found; otherwise, returns None.
    """
    # Define the mapping from shifted keys to their unshifted counterparts
    mappings = {
        '~': '`',   # Tilde and grave accent (backtick)
        '!': '1',   # Exclamation mark and number 1
        '@': '2',   # Commercial at and number 2
        '#': '3',   # Pound sign and number 3
        '$': '4',   # Dollar sign and number 4
        '%': '5',   # Percentage symbol and number 5
        '^': '6',   # Circumflex accent and number 6
        '&': '7',   # Ampersand and number 7
        '*': '8',   # Asterisk and number 8
        '(' : '9',  # Left parenthesis and number 9
        ')': '0',   # Right parenthesis and number 0
        '_': '-',   # Underscore and equal sign
        '+': '=',   # Plus sign and dash/hyphen
        '{': '[',   # Left curly brace and left square bracket
        '}': ']',   # Right curly brace and right square bracket
        '|': '\\',  # Vertical bar and backslash
        ':': ';',   # Colon and semicolon
        '"': "'",   # Double quote and single quote
        '<': ',',   # Less than sign and comma
        '>': '.',   # Greater than sign and period
        '?': '/',   # Question mark and forward slash
    }

    mapped_key = mappings.get(key, key)
    return mapped_key


# Read recorded keystrokes (labels)
# tsv format: is pressed (1 for press, 0 for release), timestamp in ms, key
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

        is_press, timestamp_str, event_key = line.split('\t')
        timestamp = int(timestamp_str)

        event_key = re.sub(r'Key\.', '', event_key)
        event_key = re.sub(r'^\'|\'$', '', event_key)
        event_key = event_key.lower()

        if event_key == '"\'"':
            event_key = "'"

        event_key = map_shifted_keys(event_key)

        if is_press == '1':
            key_events.append({'key': event_key, 'start': timestamp, 'end': timestamp + 200})
            continue
        if is_press == '0':
            for event in key_events[i:]:
                if event['key'] == event_key:
                    # Only sets the end time if the timestamp is less than 200ms after the start time
                    if timestamp < event['end']:
                        event['end'] = timestamp
                    i += 1
                    break

    label_file.close()
    return key_events


def load_data(label_file="labels.tsv", audio_file="audio.wav"):
    audio, sample_rate = load_audio(audio_file)
    cleaned_audio = reduce_noise(audio, sample_rate)
    
    labels = None
    if label_file:
        labels = read_labels(label_file)

    return cleaned_audio, labels, sample_rate

# Segment audio based on labels
# ms_pad: amount of time (in ms) before press to include
def labeled_audio_segmentation(labels, audio, sr=44100, ms_pad=20):
    keystrokes = [] # list of segments of audio
    labels_list = [] # list of dicts that contain key pressed, time pressed, time released
    padding = int(sr * (ms_pad / 1000.))
    for event in labels:

        label = event['key']
        start = event['start'] - padding

        end = start + int(sr * SEGMENT_LEN)

        key_audio = audio[start:end]
        keystrokes.append(key_audio)
        labels_list.append(label)

    return keystrokes, labels_list

# Indentifies keystrokes using detector model and segments them
def unlabeled_audio_segmentation(audio, sr=44100, confidence_threshold=0.9):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loads model and transformers
    model = load_model('detector')
    transformers = load_transformers('model_data/detector/transformer_dumps/')

    keystrokes = []
    keystroke_length = int(SEGMENT_LEN * sr)

    # iterate over 20ms chunks of audio
    chunk_size = int(sr * 0.02) # 20ms

    i = 0
    while i < len(audio):
        chunk_end = min(i + chunk_size, len(audio))
        chunk = audio[i:chunk_end]

        # pad chunk if it's on the end of the file
        if len(chunk) < chunk_size:
            diff = chunk_size - len(chunk)
            chunk = np.pad(chunk, (0, diff), 'constant', constant_values=(0))

        # transform chunk
        chunk_array = convert_to_array([chunk], n_fft=64)
        transformed_chunk, _ = transform_data(chunk_array, None, transformers)

        # run detector on chunk
        # TODO batch chunks before running model on them
        pred = model(torch.tensor(transformed_chunk.astype(np.float32)).to(device)).cpu().detach().numpy()
        # only considers it positive if the model confidence is over the threshold
        pred_int = decode_binary_label(pred[0], confidence_threshold)
        if pred_int == 1:
            # get keystroke segment (200ms)
            keystroke = audio[i:i + keystroke_length]
            keystrokes.append(keystroke)

            # skip past part of keystroke
            #i += chunk_size * 4
        
        # go to next chunk
        i += chunk_size
        
    return keystrokes

def extract_features(keystroke, n_fft):
    D = librosa.stft(keystroke, n_fft=n_fft)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    spectrogram = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db))
    return spectrogram.flatten()


def convert_to_array(list_of_keys: list, n_fft=512):
    list_of_arrays = []
    for key in list_of_keys:
        key_array = np.array(key)
        list_of_arrays.append(extract_features(key_array, n_fft))

    return list_of_arrays

def scale_features(data: pd.DataFrame):
    sc = StandardScaler()
    sc = sc.fit(data)
    return sc.transform(data), sc

# Reduce dimensions from >1000 to n_components with PCA
def dim_reduction(data: pd.DataFrame, n_components=128):
    pca = PCA(n_components=n_components)
    pca = pca.fit(data)
    return pca.transform(data), pca

# Fit one-hot encoder for labels
def one_hot(labels):
    labels_reshaped = np.array(labels).reshape(-1, 1)

    encoder = OneHotEncoder()
    encoder = encoder.fit(labels_reshaped)
    return encoder

def preprocess_data(data_dir='data', labeled=True):
    base_names = []
    extensions = ['.txt', '.wav'] if labeled else ['.wav']
    filenames = os.listdir(data_dir)

    for file in filenames:
        base, ext = os.path.splitext(file)
        # Appends file base name to base_names if it has one of the two extensions and is not already in base_names
        if ext in extensions and base not in base_names:
            base_names.append(base)

    # loads the data from every pair of txt and wav files in the data_dir    
    dataframe = pd.DataFrame()

    for base in base_names:
        print(f'Processing {base} file(s)...', end='')
        label_file = data_dir + '/' + base + '.tsv'
        audio_file = data_dir + '/' + base + '.wav'

        if labeled:
            audio, labels, sr = load_data(label_file, audio_file)
            segmented_audio, seg_labels = labeled_audio_segmentation(labels, audio, sr)

            key_df = pd.DataFrame(convert_to_array(segmented_audio))
            key_df['label'] = seg_labels

            dataframe = pd.concat([dataframe, key_df], ignore_index=True)
        
        else:
            audio, _, sr = load_data(label_file=None, audio_file=audio_file)
            segmented_audio = unlabeled_audio_segmentation(audio, sr)

            key_df = pd.DataFrame(convert_to_array(segmented_audio))
            dataframe = pd.concat([dataframe, key_df], ignore_index=True)
        print(" Finished.")
            
    
    if labeled:
        features = dataframe.drop('label', axis=1)
        labels = dataframe['label']
    else:
        features = dataframe
    features.fillna(0, inplace=True)

    if labeled:
        scaled_features, scaler = scale_features(features)
        reduced_features, pca = dim_reduction(scaled_features)

        ohe = one_hot(labels)

        processed_df = pd.DataFrame(reduced_features)

        return processed_df, labels, {'scaler': scaler, 'pca': pca, 'encoder': ohe}
    else:
        # TODO add transforming of audio data using saved model here
        return features, None, None


def main():
    df, labels, _ = preprocess_data()
    print(df.head())
    print(df.shape)

if __name__ == '__main__':
    main()