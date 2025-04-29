import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from shutil import rmtree
from preprocess_data import preprocess_data
from deep_learning_functions import *
from models import ClassificationModel, CustomDataset
from torch.utils.data import DataLoader

class EarlyStopping():
    def __init__(self, patience=5, n_epochs=3):
        self.patience = patience
        self.curr_patience = 0
        self.n_epochs = n_epochs
    
    def over_patience(self) -> bool:
        return self.curr_patience >= self.patience 
    
    # Increments curr_patience if the current loss is less than the mean of the previous epochs in range
    def step(self, history: list, curr_epoch: int):
        if curr_epoch < self.n_epochs-1:
            self.curr_patience = 0
            return
        
        curr_loss = history[curr_epoch]
        past_loss = history[(curr_epoch - self.n_epochs-1):curr_epoch+1]
        print(curr_loss, past_loss)
        if curr_loss >= np.mean(past_loss):
            self.curr_patience += 1
        else:
            self.curr_patience = 0
        return


def train_model(data_dir='data', epochs=10, batch_size=256, return_model=True, save_model=True, save_dir='model_data/', delete_existing_model=True, force_preprocess=True):

    # Deletes model_data directory
    if save_model and delete_existing_model and os.path.exists(save_dir):
        user_input = input(f"Warning: All files in {save_dir} will be deleted! Are you sure that you want to continue? [y/N] ")
        if user_input.lower() == 'y':
            print("Directory will be deleted after training is completed.")
        else:
            print("Files will not be deleted. Going back...")
            return

    if not os.path.exists(data_dir + "/imgs") or force_preprocess:
        print("Converting audio data to images...")
        transformers = preprocess_data(data_dir=data_dir, del_img_dir=True)
    else:
        transformers = load_transformers(data_dir + "/imgs/")
    print("Data loading complete!")

    train_dataset = CustomDataset(data_dir, annotation_file="train.csv", encoder=transformers['encoder'])
    test_dataset = CustomDataset(data_dir, annotation_file="test.csv", encoder=transformers['encoder'])
    
    # Uses GPU if available, otherwise uses CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_shape = train_dataset.get_data_shape()
    output_size = train_dataset.get_label_shape()
    hidden_sizes = [128, 64] # Hidden layer sizes

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = ClassificationModel(input_shape, hidden_sizes, output_size).to(device)
    l = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=3e-4)

    ## Training model
    torch.manual_seed(1)
    np.random.seed(1)

    train_loss = [None]*epochs
    val_loss = [None]*epochs

    train_acc = [None]*epochs
    val_acc = [None]*epochs
    final_acc = 0.

    early_stop = EarlyStopping(patience=2, n_epochs=3)

    for epoch in range(epochs):
        tr_loss_sum = 0.
        tr_acc_sum = 0.
        tr_batch_num = 0
        model.train()

        for X, y in train_dataloader:

            pred = model(X)
            loss = l(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            curr_loss = loss.item()
            tr_loss_sum += curr_loss

            predictions = torch.argmax(pred, dim=1)
            truth = torch.argmax(y, dim=1)
            curr_acc = (predictions == truth).float().mean().cpu()
            tr_acc_sum += curr_acc

            tr_batch_num += 1
            if tr_batch_num % 50 == 0:
                print(f"Training - Epoch {epoch + 1} - Batch {tr_batch_num} - Loss: {tr_loss_sum / tr_batch_num :.2f} - Accuracy: {tr_acc_sum / tr_batch_num * 100 :.2f}%", end='\r')

        tr_loss = tr_loss_sum / tr_batch_num
        tr_acc = tr_acc_sum / tr_batch_num

        model.eval()

        te_loss_sum = 0.
        te_acc_sum = 0.
        te_batch_num = 0
        
        for X, y in test_dataloader:
            pred = model(X)
            curr_loss = l(pred, y).item()
            te_loss_sum += curr_loss

            predictions = torch.argmax(pred, dim=1)
            truth = torch.argmax(y, dim=1)
            curr_acc = (predictions == truth).float().mean().cpu()
            te_acc_sum += curr_acc

            te_batch_num += 1
            if te_batch_num % 50 == 0:
                print(f"Testing - Epoch {epoch + 1} - Batch {te_batch_num} - Loss: {te_loss_sum / te_batch_num :.2f} - Accuracy: {te_acc_sum / te_batch_num * 100 :.2f}%", end='\r')
        
        te_loss = te_loss_sum / te_batch_num
        te_acc = te_acc_sum / te_batch_num

        train_loss[epoch] = tr_loss
        val_loss[epoch] = te_loss
        train_acc[epoch] = tr_acc
        val_acc[epoch] = te_acc
        final_acc = int(te_acc * 100)

        #early_stop.step(val_loss, epoch)
        
        #if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'Epoch {epoch+1} - train loss: {tr_loss :.4f} - train acc: {tr_acc * 100:.2f}% - val loss: {te_loss :.4f} - val acc: {te_acc * 100:.2f}%')
        
        # If there has been no improvement for at least 'patience' epochs
        #if early_stop.over_patience():
        #    print(f"Stopping early at epoch {epoch+1} with a validation accuracy of {final_acc}%...")
        #    break

    if save_model:
        # Delete directory and make a new one
        rmtree(save_dir)
        print('Directory deleted')
        os.makedirs(save_dir)

        # Save transformer models
        dump_transformers(transformers, dir=save_dir)

        # Save model layer sizes to file
        write_model_info(input_shape, hidden_sizes, output_size, dir=save_dir)

        # Save model state dict to file
        torch.save(model.state_dict(), f'{save_dir}/model.pt')
        print(f"Successfully saved model with a validation accuracy of {final_acc}%.")

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