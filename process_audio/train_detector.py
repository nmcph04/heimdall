import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import os
from shutil import rmtree
from detector_preprocess import preprocess_for_detector
from deep_learning_functions import *
from models import DetectorModel

def train_detector(data_dir='data', epochs=1_000, return_model=True,
                   save_model=True, save_dir='model_data/', delete_existing_model=True, auto_rm=False):
    save_dir = save_dir + 'detector/'

    # Deletes model_data directory
    if save_model and delete_existing_model and os.path.exists(save_dir):
        if not auto_rm:
            user_input = input(f"Warning: All files in {save_dir} will be deleted! Are you sure that you want to continue? [y/N] ")
            if user_input.lower() == 'y':
                rmtree(save_dir)
                print('Directory deleted')
            else:
                print("Files will not be deleted. Exiting program...")
                exit(0)
        else:
            rmtree(save_dir)
            print(f'{save_dir} directory deleted')

    features, labels, transformers = preprocess_for_detector(data_dir=data_dir)

    # Oversample and augment dataset
    X, y = detector_oversample(features, labels)
    X, y = augment_data(X, y, 3.)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Uses GPU if available, otherwise uses CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert datasets to tensors
    X_train = torch.tensor(X_train.astype(np.float32)).to(device)
    y_train = torch.tensor(y_train.reshape(-1, 1).astype(np.float32)).to(device)
    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test.reshape(-1, 1).astype(np.float32)).to(device)

    input_size = X_train.shape[1]
    output_size = 1 # Binary output
    hidden_size = [128, 64, 32] # Hidden layer sizes
    lr = 0.001

    model = DetectorModel(input_size, hidden_size, output_size).to(device)
    l = nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.00001, momentum=0.3)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    ## Training model

    torch.manual_seed(0)
    np.random.seed(0)


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
        scheduler.step()
        tr_loss = loss.item()
        tr_acc = detector_accuracy(pred, y_train)

        model.eval()

        pred = model(X_test)
        te_loss = l(pred, y_test).item()
        te_acc = detector_accuracy(pred, y_test)


        train_loss[epoch] = tr_loss
        val_loss[epoch] = te_loss
        train_acc[epoch] = tr_acc
        val_acc[epoch] = te_acc
        final_acc = int(te_acc * 100)
        if (epoch+1) % 50 == 0 or epoch == 0:
            print(f'Epoch {epoch+1} - train loss: {tr_loss :.4f} - val loss: {te_loss :.4f} - val acc: {te_acc:.4f}')
    
    if save_model and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_model:
        # Save transformer models
        dump_transformers(transformers, dir=save_dir)

        # Save model layer sizes to file
        write_model_info(input_size, hidden_size, output_size, dir=save_dir)

        # Save model state dict to file
        torch.save(model.state_dict(), f'{save_dir}/model.pt')
        print(f"Successfully saved model with a validation accuracy of {final_acc}% after {epochs} epochs.")

    trainhist = pd.DataFrame({'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'epoch': np.arange(epochs)})
    
    if save_model:
        trainhist.to_csv(save_dir + 'trainhist.csv')

    if return_model:
        return model, trainhist

def main():
    train_detector(return_model=False)

if __name__ == '__main__':
    main()