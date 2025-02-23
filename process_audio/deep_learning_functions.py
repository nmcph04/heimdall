import numpy as np
from imblearn.over_sampling import RandomOverSampler
import torch
import os
from pickle import dump

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