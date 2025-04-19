import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from shutil import rmtree
from preprocess_data import preprocess_data
from deep_learning_functions import *
from models import ClassificationModel

# Returns True if the current loss is less than the mean of the previous epochs in range
def no_improvement(history: list, curr_epoch: int, early_stop_n: int) -> bool:
    if curr_epoch < early_stop_n-1:
        return False
    curr_loss = history[curr_epoch]
    past_loss = history[(curr_epoch - early_stop_n-1):curr_epoch+1]
    if curr_loss >= np.mean(past_loss):
        return True
    else:
        return False

def train_model(data_dir='data', epochs=5_000, return_model=True, save_model=True, save_dir='model_data/', delete_existing_model=True):

    # Deletes model_data directory
    if save_model and delete_existing_model and os.path.exists(save_dir):
        user_input = input(f"Warning: All files in {save_dir} will be deleted! Are you sure that you want to continue? [y/N] ")
        if user_input.lower() == 'y':
            rmtree(save_dir)
            print('Directory deleted')
        else:
            print("Files will not be deleted. Exiting program...")
            exit(0)

    X_train, X_test, y_train, y_test, transformers = preprocess_data(data_dir=data_dir)
    print("Data loading complete!")

    # Uses GPU if available, otherwise uses CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert datasets to tensors
    X_train = torch.tensor(X_train.astype(np.float32)).to(device)
    y_train = torch.tensor(y_train.astype(np.float32)).to(device)
    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    hidden_sizes = [512, 256, 128] # Hidden layer sizes

    model = ClassificationModel(input_size, hidden_sizes, output_size).to(device)
    l = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=3e-4)

    ## Training model

    torch.manual_seed(1112)
    np.random.seed(1112)


    train_loss = [None]*epochs
    val_loss = [None]*epochs

    train_acc = [None]*epochs
    val_acc = [None]*epochs
    final_acc = 0.

    EARLY_STOPPING_N = 40

    for epoch in range(epochs):
        model.train()

        pred = model(X_train)
        loss = l(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss = loss.item()

        predictions = torch.argmax(pred, dim=1)
        truth = torch.argmax(y_train, dim=1)
        tr_acc = (predictions == truth).float().mean().cpu()

        model.eval()

        pred = model(X_test)
        te_loss = l(pred, y_test).item()

        predictions = torch.argmax(pred, dim=1)
        truth = torch.argmax(y_test, dim=1)
        te_acc = (predictions == truth).float().mean().cpu()

        train_loss[epoch] = tr_loss
        val_loss[epoch] = te_loss
        train_acc[epoch] = tr_acc
        val_acc[epoch] = te_acc
        final_acc = int(te_acc * 100)
        #if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'Epoch {epoch+1} - train loss: {tr_loss :.4f} - train acc: {tr_acc:.4f} - val loss: {te_loss :.4f} - val acc: {te_acc:.4f}')
        
        if epoch > 1 and no_improvement(val_loss, epoch, EARLY_STOPPING_N):
            print(f"Model has not improved for {EARLY_STOPPING_N} epochs. Stopping early at epoch {epoch+1} with an accuracy of {final_acc}%...")
            break
    
    if save_model and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_model:
        # Save transformer models
        dump_transformers(transformers, dir=save_dir)

        # Save model layer sizes to file
        write_model_info(input_size, hidden_sizes, output_size, dir=save_dir)

        # Save model state dict to file
        torch.save(model.state_dict(), f'{save_dir}/model.pt')
        print(f"Successfully saved model with a validation accuracy of {final_acc}% after {epochs} epochs.")

    trainhist = pd.DataFrame({'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'epoch': np.arange(epochs)})
    
    if save_model:
        trainhist.to_csv(save_dir + 'trainhist.csv')

    if return_model:
        return model, trainhist, transformers

def main():
    train_model(return_model=False)

if __name__ == '__main__':
    main()