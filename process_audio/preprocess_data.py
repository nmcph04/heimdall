import pandas as pd
import numpy as np
import noisereduce as nr
import os
from shutil import rmtree
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import librosa
from deep_learning_functions import oversample_dataset, augment_data
from PIL import Image
from pickle import dump

# Loads data, reduces noise, extracts features, standardizes, and reduces the dimensionality of the data

SEGMENT_LEN = 0.2 # each segment should be 200ms long

def load_audio(filename):
    return librosa.load(filename, sr=None)

def reduce_noise(audio, sample_rate):
    reduced_noise = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.5)
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

# Segments audio into equal length chunks, moving 'step' samples forward every time
def naive_segmentation(audio: np.ndarray, sr=44100, seg_len=0.2, step=60):
    seg = []
    sample_length = int(seg_len * sr)
    silent_num = 0
    for i in range(0, len(audio), step):
        segment = audio[i: i + sample_length]

        if is_quiet(segment, 0.0005):
            silent_num += 1
            if silent_num > 50:
                continue
        elif silent_num != 0:
            silent_num = 0


        if len(segment) != sample_length:
            pad_amt = sample_length - len(segment)
            segment = np.pad(segment, (0, pad_amt), 'constant', constant_values=(0))

        seg.append(segment)
    
    return np.array(seg)


def is_quiet(chunk: list, threshold: float) -> bool:
    if np.max(np.abs(chunk)) < threshold:
        return True
    else:
        return False


def extract_features(keystroke, n_fft):
    D = librosa.stft(keystroke, n_fft=n_fft)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Scale data from [0-1]
    spectrogram = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db))

    # Scale data to [0,255] so it can be saved as an image
    spectrogram = (spectrogram * 255).astype(np.uint8)

    return spectrogram


def convert_to_array(list_of_keys: list, n_fft=256):
    list_of_arrays = []
    for key in list_of_keys:
        key_array = np.array(key)
        list_of_arrays.append(extract_features(key_array, n_fft))

    return np.array(list_of_arrays)

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

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
    transformed = encoder.fit_transform(labels_reshaped)
    return transformed, encoder

# All classes with less than x samples will be reduced into a single class
def reduce_small_classes(labels, threshold=10):
    reduced_class_label = "□"

    # Get classes to reduce
    classes_to_reduce = []
    class_names, class_counts = np.unique(labels, return_counts=True)
    for i in range(len(class_names)):
        count = class_counts[i]
        label = class_names[i]
        
        if count <= threshold:
            classes_to_reduce.append(label)
    
    reduced_labels = labels.copy()
    # Get all indices of the elements to reduce
    for i in range(len(labels)):
        if labels[i] in classes_to_reduce:
            reduced_labels[i] = reduced_class_label
    
    return np.array(reduced_labels)

# Saves the images to directory at path/category/base_name and returns a DataFrame mapping file names to labels
# Category should be either 'train' or 'test'
def save_images(features: np.ndarray, labels: np.ndarray, path: str, base_name: str, category: str) -> pd.DataFrame:
    if category not in ['train', 'test']:
        raise ValueError

    path = path + "/imgs/"
    if not os.path.exists(path):
        os.mkdir(path)
    
    path = path + category + "/"
    if not os.path.exists(path):
        os.mkdir(path)

    dir_path = path + base_name + "/"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    map_df = pd.DataFrame(columns=['img_name', 'label'])
    i = 1
    for X, y in zip(features, labels):
        curr_filename = dir_path + f"{i :06d}" + ".jpeg"
        if y[0] is not None:
            Image.fromarray(X).save(curr_filename)
            map_df.loc[len(map_df)] = [curr_filename, y[0]]
        i += 1
    return map_df

# Finds the WAV file with the greatest size and loads it in for fitting preprocessing transformers
# Removes this file from the base_names list
# Return the X_train, X_test, y_train, y_test data along with a dict of fitted transformers
def load_largest_file(path: str, base_names: list): 
    transformers = {}

    sizes = {}
    path = path + "/"
    for i in range(len(base_names)):
        sizes[i] = os.path.getsize(path + base_names[i] + ".wav")

    # Gets the base name with the largest WAV file, removes it from the list of bases
    largest_file_idx = max(sizes, key=sizes.get)
    largest_base = base_names[largest_file_idx]
    del base_names[largest_file_idx]

    X, y = load_files(path, largest_base)

    #y = reduce_small_classes(y)
    y, transformers['encoder'] = one_hot(y)

    X_train, X_test, y_train, y_test = transform_data(X, y, None)

    y_train = transformers['encoder'].inverse_transform(y_train)
    y_test = transformers['encoder'].inverse_transform(y_test)

    train_df = save_images(X_train, y_train, path=path, base_name=largest_base, category='train')
    test_df = save_images(X_test, y_test, path=path, base_name=largest_base, category='test')

    print(" Finished.", flush=True)

    return train_df, test_df, transformers


# Loads in the label and feature files that share base_name at path
def load_files(path: str, base_name: str) -> np.ndarray | np.ndarray:
    print(f'Processing {base_name} files...', end='', flush=True)
    label_file = path + '/' + base_name + '.tsv'
    audio_file = path + '/' + base_name + '.wav'
    
    audio, labels, sr = load_data(label_file, audio_file)
    segmented_audio, seg_labels = labeled_audio_segmentation(labels, audio, sr)
    
    X = segmented_audio
    y = seg_labels

    return X, y

# encodes labels, converts data to spectrograms and shuffles it
def transform_data(features, labels, transformers: dict):
    if transformers:
        y = transformers['encoder'].transform(np.array(labels).reshape(-1, 1))
    else:
        y = labels

    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=1)

    # Oversample and augment data
    X_train, y_train = oversample_dataset(X_train, y_train)
    #X_train, y_train = augment_data(X_train, y_train, 3.)

    # Convert raw audio to spectrograms
    X_train = convert_to_array(X_train)
    X_test = convert_to_array(X_test)

    # Shuffle training data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    X_train = X_train[indices]
    y_train = y_train[indices]

    if transformers:
        y_train = transformers['encoder'].inverse_transform(y_train)
        y_test = transformers['encoder'].inverse_transform(y_test)

    return X_train, X_test, y_train, y_test


def preprocess_data(data_dir='data/', del_img_dir=False):
    # Delete an existing image directory
    if del_img_dir:
        try:
            rmtree(data_dir + "/imgs/")
        except FileNotFoundError:
            pass

    base_names = []
    extensions = ['.tsv', '.wav']
    filenames = os.listdir(data_dir)

    for file in filenames:
        base, ext = os.path.splitext(file)
        # Appends file base name to base_names if it has one of the two extensions and is not already in base_names
        if ext in extensions and base not in base_names:
            base_names.append(base)
    base_names.sort()

    # Processes largest file and fits transformers with it
    train_df, test_df, transformers = load_largest_file(data_dir, base_names)

    # Loads and processes the rest of the files with the fitted transformers
    for base in base_names:
        X, y = load_files(data_dir, base)
        X_train, X_test, y_train, y_test = transform_data(X, y, transformers)

        train_df_new = save_images(X_train, y_train, path = data_dir, base_name=base, category='train')
        test_df_new = save_images(X_test, y_test, path = data_dir, base_name=base, category='test')
        train_df = pd.concat([train_df, train_df_new], ignore_index=True)
        test_df = pd.concat([test_df, test_df_new], ignore_index=True)

        print(" Finished.", flush=True)
    
    # Save maps to file
    train_df.to_csv(data_dir + "/imgs/train.csv", index=False)
    test_df.to_csv(data_dir + "/imgs/test.csv", index=False)

    # Save transformers to file
    for name, transformer in transformers.items():
        path = data_dir + "/imgs/" + name + '.pkl'
        dump(transformer, open(path, 'wb'))

    return transformers


def main():
    df, labels, _ = preprocess_data()
    print(df.head())
    print(df.shape)

if __name__ == '__main__':
    main()