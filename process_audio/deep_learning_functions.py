import numpy as np
from scipy.signal import resample
import pandas as pd
from imblearn.over_sampling import SMOTE
import torch
import os
from pickle import dump, load
from models import DetectorModel, ClassificationModel

def detector_oversample(features, labels):
    oversampler = SMOTE()
    over_X, over_y = oversampler.fit_resample(features, labels)
    return over_X, over_y

def oversample_dataset(features, labels, encoder):
    # Gets number of unique label elements
    label_shape = labels.shape[1]

    # Converts labels to integers for SMOTE to process
    encoded_labels = list(int(np.argmax(x)) for x in labels)

    oversampler = SMOTE()
    over_X, over_y = oversampler.fit_resample(features, encoded_labels)

    # Convert integer labels back into class names
    zero_array = np.zeros((len(over_y), label_shape))
    for i, encoded in enumerate(over_y):
        zero_array[i][encoded] = 1

    return np.array(over_X), zero_array

def time_stretch(audio, stretch_factor, target_length=17733):
    if stretch_factor <= 0:
        raise ValueError("Stretch factor must be greater than 0")
    
    # Stretch or compress the audio
    stretched_audio = resample(audio, int(len(audio) / stretch_factor))
    
    # Resample to the target length
    resized_audio = resample(stretched_audio, target_length)
    return resized_audio


def pitch_shift(audio, n_steps, sample_rate=44100, target_length=17733):
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
        # multiplies X values by random number from .8 to 1.2
        noise = (rng.random(X.shape[1]) / 5 - 0.1) * 2 + 1
        synthesized_X[i] = X[idx] * noise
        synthesized_y[i] = y[idx]
        i += 1
    
    # Time stretch/compress n random samples
    stretch_factor = (rng.random(n_per_method) / 5 - 0.1) * 2 + 1
    for j, idx in enumerate(random_indices[1]):
        # stretches X values by a factor .8x to 1.2x
        synthesized_X[i] = time_stretch(X[idx], stretch_factor[j])
        synthesized_y[i] = y[idx]
        i += 1

    # Pitch shift n random samples
    shift_factor = (rng.random(n_per_method) / 5 - 0.1) * 2 + 1
    for j, idx in enumerate(random_indices[2]):
        # shifts pitch of X values by .8x to 1.2x
        synthesized_X[i] = pitch_shift(X[idx], shift_factor[j])
        synthesized_y[i] = y[idx]
        i += 1
    
    
    return np.concatenate((X, synthesized_X), axis=0), np.concatenate((y, synthesized_y), axis=0)

# Decodes binary labels encoded with one-hot encoding
def decode_binary_label(label:np.ndarray, threshold=0.9):
    return 1 if label[1] >= threshold else 0

# Get accuracy for the detector model
def detector_accuracy(pred_y: torch.Tensor, true_y: torch.Tensor):
    # Get binary prediction from sigmoid output
    pred_y = pred_y.cpu().detach().numpy()
    pred_y_binary = (pred_y >= 0.5).astype(np.float32)

    true_y = true_y.cpu().detach().numpy()

    num_values = np.float32(pred_y_binary.shape[0])
    num_correct = np.sum(pred_y_binary == true_y)
    
    return num_correct / num_values

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
    file.write(str(input_layer) + '\n')

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
        input_layer = int(file.readline().strip())
        hidden_layers = [int(x) for x in file.readline().strip().split(',') if x.strip()]
        output_layer = int(file.readline().strip())
    
    return input_layer, hidden_layers, output_layer 

def feature_pipeline(transformers, features):
    scaled = transformers['scaler'].transform(features)
    return transformers['pca'].transform(scaled)

def label_ohe(encoder, labels):
    return encoder.transform(labels.reshape(-1, 1))

def transform_data(X, y, transformers: dict):
    features = feature_pipeline(transformers, X)
    labels = None
    if y:
        labels = label_ohe(transformers['encoder'], np.array(y))

    return features, labels

# Load data transformers
def load_transformers(dir='model_data/classifier/transformer_dumps/'):
    encoder = None
    try:
        encoder = load(open(dir + 'encoder.pkl', 'rb'))
    except:
        pass
    scaler = load(open(dir + 'scaler.pkl', 'rb'))
    pca = load(open(dir + 'pca.pkl', 'rb'))
    if encoder: 
        return {'encoder': encoder, 'scaler': scaler, 'pca': pca}
    else:
        return {'scaler': scaler, 'pca': pca}

# dir is the directory that has both the classifier and detector directories within it
# Model type must be either 'classifier' or 'detector'
def load_model(model_type: str, path='model_data/'):
    is_classfier = False
    if model_type.lower() == 'classifier':
        path += '/classifier/'
        is_classfier = True
    elif model_type.lower() == 'detector':
        path += '/detector/'
    else:
        raise Exception(f"model_type is {model_type}, but must be either 'classifier' or 'detector'!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size, hidden_sizes, output_size = read_model_info(path)
    
    if is_classfier:
        model = ClassificationModel(input_size, hidden_sizes, output_size).to(device)
    else:
        model = DetectorModel(input_size, hidden_sizes, output_size).to(device)

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