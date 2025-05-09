import numpy as np
from scipy.signal import resample
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
import torch
import os
from pickle import dump, load
from models import ClassificationModel


def oversample_dataset(features, labels):
    features = np.array(features)
    labels = np.array(labels)

    # Converts labels to integers for SMOTE to process
    label_shape = labels.shape[1]
    encoded_labels = list(int(np.argmax(x)) for x in labels)

    # Uses random oversampling so that all classes have at least 10 samples
    at_least_ten = {}
    elements, element_counts = np.unique(encoded_labels, return_counts=True)
    for k, v in zip(elements, element_counts):
        at_least_ten[int(k)] = max(10, int(v))

    ros = RandomOverSampler(sampling_strategy=at_least_ten)
    over_X, over_y = ros.fit_resample(features, encoded_labels)

    # Uses SMOTE to finish the oversampling
    smote = SMOTE(sampling_strategy='not majority')
    over_X, over_y = smote.fit_resample(over_X, over_y)

    # Convert integer labels back into class names
    zero_array = np.zeros((len(over_y), label_shape))
    for i, encoded in enumerate(over_y):
        zero_array[i][encoded] = 1

    return over_X, zero_array

def time_stretch(audio, stretch_factor, target_length=8820):
    if stretch_factor <= 0:
        raise ValueError("Stretch factor must be greater than 0")
    
    # Stretch or compress the audio
    stretched_audio = resample(audio, int(len(audio) / stretch_factor))
    
    # Resample to the target length
    resized_audio = resample(stretched_audio, target_length)
    return resized_audio


def pitch_shift(audio, n_steps, sample_rate=44100, target_length=8820):
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    
    # Calculate the pitch shift factor
    pitch_factor = 2 ** (n_steps / 12.0)  # 12 steps = 1 octave
    
    # Stretch or compress the audio in time
    shifted_audio = resample(audio, int(len(audio) / pitch_factor))
    
    # Resample back to the target length
    resized_audio = resample(shifted_audio, target_length)
    return resized_audio

# Perform data augmentation to increase dataset size
def augment_data(X: np.ndarray, y: np.ndarray, pct_added: float):
    initial_dataset_size = X.shape[0]
    n_added = int(initial_dataset_size * pct_added - initial_dataset_size)
    # Number of new samples from each method
    n_per_method = int(n_added / 3)

    # Pre-allocate memory for synthesized data
    synthesized_X = np.empty((n_per_method * 3, X.shape[1]), dtype=X.dtype)
    synthesized_y = np.empty((n_per_method * 3, y.shape[1]), dtype=y.dtype)

    # Generate n random indices from (0, size of dataset]
    rng = np.random.default_rng()
    random_indices = rng.integers(0, initial_dataset_size, (3, n_per_method))

    i = 0
    # Add noise to n random samples
    for idx in random_indices[0]:
        # multiplies X values by random number from 0.5 to 1.5
        noise = (rng.random(X.shape[1]) + 0.5)
        synthesized_X[i] = X[idx] * noise
        synthesized_y[i] = y[idx]
        i += 1
    
    # Time stretch/compress n random samples
    stretch_factor = (rng.random(n_per_method) + 0.5)
    for j, idx in enumerate(random_indices[1]):
        # stretches X values by a factor 0.5x to 1.5x
        synthesized_X[i] = time_stretch(X[idx], stretch_factor[j])
        synthesized_y[i] = y[idx]
        i += 1

    # Pitch shift n random samples
    shift_factor = (rng.random(n_per_method) / + 0.5)
    for j, idx in enumerate(random_indices[2]):
        # shifts pitch of X values by 0.5x to 1.5x
        synthesized_X[i] = pitch_shift(X[idx], shift_factor[j])
        synthesized_y[i] = y[idx]
        i += 1
    
    
    return np.concatenate((X, synthesized_X), axis=0), np.concatenate((y, synthesized_y), axis=0)

# Decodes binary labels encoded with one-hot encoding
def decode_binary_label(label:np.ndarray, threshold=0.9):
    return 1 if label[1] >= threshold else 0

# Get accuracy
def getAcc(pred_y: torch.Tensor, true_y: torch.Tensor):
    pred_y = pred_y.cpu().detach().numpy()
    pred_y = np.argmax(pred_y, axis=1)

    true_y = true_y.cpu().detach().numpy()
    true_y = np.argmax(true_y, axis=1)

    num_values = np.float32(pred_y.shape[0])
    num_correct = np.sum(pred_y == true_y)
    
    return num_correct / num_values

# writes model layer sizes to file, so they can be used when loading the model
def write_model_info(input_layer, hidden_layers, output_layer, dir=''):
    path = dir + 'model_info.txt'
    file = open(path, 'x')
    for shape in input_layer:
        file.write(str(shape) + ',')
    
    file.write('\n')

    for layer in hidden_layers:
        file.write(str(layer) + ',')

    file.write('\n' + str(output_layer))
    file.close()

# dumps the data transformers used to preprocess the data so they can be reused when using the model
def dump_transformers(transformers: dict, dir=''):
    dump_dir = dir + 'transformer_dumps/'

    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    for name, transformer in transformers.items():
        path = dump_dir + name + '.pkl'
        dump(transformer, open(path, 'wb'))

def read_model_info(path='model_data/'):
    with open(path + 'model_info.txt', 'r') as file:
        input_layer = [int(x) for x in file.readline().strip().split(',') if x.strip()]
        hidden_layers = [int(x) for x in file.readline().strip().split(',') if x.strip()]
        output_layer = int(file.readline().strip())
    
    return input_layer, hidden_layers, output_layer 

def label_ohe(encoder, labels):
    return encoder.transform(labels.reshape(-1, 1))

# Load data transformers
def load_transformers(dir='model_data/transformer_dumps/'):
    encoder = load(open(dir + 'encoder.pkl', 'rb'))
    return {'encoder': encoder}

# path is the directory that has the model directory within it
def load_model(path='model_data/'):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size, hidden_sizes, output_size = read_model_info(path)
    
    model = ClassificationModel(input_size, hidden_sizes, output_size).to(device)

    model_path = path + 'model.pt'
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(device)))
    model.eval()
    return model

# emulates typing the characters in a list, printing out the result
# characters such as backspace, space, shift are handled
def emulate_typing(chars: list):
    shifted = False
    buffer = []

    for char in chars:
        if char == 'backspace':
            if buffer:
                buffer.pop()
        elif char == 'space':
            buffer.append(' ')
        elif char == 'shift' or char == 'shift_r':
            shifted = True
        elif char == 'enter':
            buffer.append('\n')
        else:
            if shifted:
                buffer.append(char.upper())
                shifted = False
            else: 
                buffer.append(char)

    print(''.join(char for char in buffer))