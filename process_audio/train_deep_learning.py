import pandas as pd
import numpy as np
from preprocess_data import preprocess_data
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from pickle import dump
import os
from shutil import rmtree

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

# Define model
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear_sequential_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size[2], output_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        logits = self.linear_sequential_stack(x)
        return logits

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

def train_model(data_dir='data', epochs=5_000, return_model=True, save_model=True, save_dir='model_data/', delete_existing_model=True):

    # Deletes model_data directory
    if delete_existing_model and os.path.exists(save_dir):
        user_input = input(f"Warning: All files in {save_dir} will be deleted! Are you sure that you want to continue? [Y/n] ")
        if user_input.lower() == 'y':
            rmtree(save_dir)
            print('Directory deleted')
        else:
            print("Files will not be deleted. Exiting program...")
            exit(0)

    features, (labels, labels_ohe), transformers = preprocess_data()

    # Oversample and augment dataset
    X, y = oversample_datset(features.to_numpy(), labels)
    X, y = augment_data(X, y, 3.)
    y = transformers['encoder'].transform(np.array(y).reshape(-1, 1)).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Uses GPU if available, otherwise uses CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert datasets to tensors
    X_train = torch.tensor(X_train.astype(np.float32)).to(device)
    y_train = torch.tensor(y_train.astype(np.float32)).to(device)
    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    hidden_size = [256, 128, 64] # Hidden layer sizes

    model = Model(input_size, hidden_size, output_size).to(device)
    l = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0012, weight_decay=0.00001, momentum=0.9)

    ## Training model

    torch.manual_seed(1112)
    np.random.seed(1112)


    train_loss = [None]*epochs
    val_loss = [None]*epochs

    train_acc = [None]*epochs
    val_acc = [None]*epochs
    final_acc = 0.

    for epoch in range(epochs):
        model.train()

        pred = model(X_train)
        loss = l(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss = loss.item()
        tr_acc = getAcc(pred, y_train)

        model.eval()

        pred = model(X_test)
        te_loss = l(pred, y_test).item()
        te_acc = getAcc(pred, y_test)


        train_loss[epoch] = tr_loss
        val_loss[epoch] = te_loss
        train_acc[epoch] = tr_acc
        val_acc[epoch] = te_acc
        final_acc = int(te_acc * 100)
        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f'Epoch {epoch+1} - train loss: {tr_loss :.4f} - val loss: {te_loss :.4f} - val acc: {te_acc:.4f}')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save transformer models
    dump_transformers(transformers, dir=save_dir)

    # Save model layer sizes to file
    write_model_info(input_size, hidden_size, output_size, dir=save_dir)

    # Save model state dict to file
    torch.save(model.state_dict(), f'{save_dir}/model.pt')
    print(f"Successfully saved model with a validation accuracy of {final_acc}% after {epochs} epochs.")

    trainhist = pd.DataFrame({'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'epoch': np.arange(epochs)})
    
    trainhist.to_csv(save_dir + 'trainhist.csv')

    if return_model:
        return model, trainhist

def main():
    train_model(return_model=False)

if __name__ == '__main__':
    main()