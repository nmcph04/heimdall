import numpy as np
from imblearn.over_sampling import RandomOverSampler
import torch
import os
from pickle import dump, load
from models import DetectorModel, ClassificationModel

# Balance dataset with oversampling
def oversample_datset(X, y):
    oversample = RandomOverSampler()
    over_X, over_y = oversample.fit_resample(X, y)

    return over_X, over_y

# Perform data augmentation to increase dataset size
def augment_data(X: np.ndarray, y: np.ndarray, pct_added: float):
    initial_dataset_size = X.shape[0]
    n_added = int(initial_dataset_size * pct_added - initial_dataset_size)

    synthesized_X = []
    synthesized_y = []

    # Generate n random indices from (0, size of dataset]
    rng = np.random.default_rng()
    random_indices = rng.integers(0, initial_dataset_size, n_added)

    # add noise to n random samples
    for idx in random_indices:
        # multiplies X values by random number from .8 to 1.2
        noise = (rng.random(X.shape[1]) / 5 - 0.1) * 2 + 1
        synthesized_X.append(X[idx] * noise)
        synthesized_y.append(y[idx])
    
    synthesized_X = np.array(synthesized_X)
    synthesized_y = np.array(synthesized_y)
    
    return np.append(X, synthesized_X, axis=0), np.append(y, synthesized_y, axis=0)

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

def read_model_info(dir='model_data/'):
    with open(dir + 'model_info.txt', 'r') as file:
        input_layer = int(file.readline().strip())
        hidden_layers = [int(x) for x in file.readline().strip().split(',') if x.strip()]
        output_layer = int(file.readline().strip())
    
    return input_layer, hidden_layers, output_layer 

def feature_pipeline(transformers, features):
    scaled = transformers['scaler'].transform(features)
    return transformers['pca'].transform(scaled)

def label_ohe(encoder, labels):
    return encoder.transform(labels, sparse_output=False)

def transform_data(X, y, transformers: dict):
    features = feature_pipeline(transformers, X)
    labels = None
    if y:
        labels = label_ohe(transformers['encoder'], y)

    return features, labels

# Load data transformers
def load_transformers(dir='model_data/classifier/transformer_dumps/'):
    encoder = load(open(dir + 'encoder.pkl', 'rb'))
    scaler = load(open(dir + 'scaler.pkl', 'rb'))
    pca = load(open(dir + 'pca.pkl', 'rb'))

    return {'encoder': encoder, 'scaler': scaler, 'pca': pca}

# dir is the directory that has both the classifier and detector directories within it
# Model type must be either 'classifier' or 'detector'
def load_model(model_type: str, dir='model_data/'):
    path = dir
    is_classfier = False
    if model_type.lower() == 'classifier':
        path += 'classifier/'
        is_classfier = True
    elif model_type.lower() == 'detector':
        path += 'detector/'
    else:
        raise Exception(f"model_type is {model_type}, but must be either 'classifier' or 'detector'!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size, hidden_sizes, output_size = read_model_info(path)
    
    if is_classfier:
        model = ClassificationModel(input_size, hidden_sizes, output_size).to(device)
    else:
        model = DetectorModel(input_size, hidden_sizes, output_size).to(device)

    model_path = path + 'model.pt'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model