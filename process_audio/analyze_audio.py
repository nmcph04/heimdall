import numpy as np
import torch
from deep_learning_functions import load_model, load_transformers, emulate_typing
from preprocess_data import load_data, convert_to_array, naive_segmentation
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.features = features
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index].to(self.device)

# Filters predictions so that one character isn't predicted more than twice in a row
def filter_predictions(pred: np.ndarray) -> list:
    if np.ndim(pred) == 0:
        return []
    filtered = []
    count = 1
    last_key = ''
    for key in pred:
        if key == last_key:
            count += 1

            if count < 3:
                filtered.append(key)
        else:
            last_key = key
            count = 1
            filtered.append(key)
    return filtered

# Filters predictions by the model's confidence in that prediction
def filter_by_confidence(predictions: np.ndarray, confidence=0.6) -> np.ndarray:
    filtered_pred = []
    for pred in predictions:
        if np.max(pred) >= confidence:
            filtered_pred.append(pred)
    
    return filtered_pred

def analyze_audio(audio_file: str, models_path='model_data/'):
    print("Loading data...", end='', flush=True)
    # Load model
    classifier = load_model(path=models_path)

    # Load transformers
    transformers = load_transformers(models_path + '/transformer_dumps/')

    # Load audio
    audio, _, sr = load_data(label_file=None, audio_file=audio_file)

    audio = audio[:5000000]

    print(' Done!', flush=True)

    # Segment audio using naive segmentation
    print("Segmenting audio...", end='', flush=True)
    X = naive_segmentation(audio, sr=sr)

    # Modify and transform the segmented audio
    X = convert_to_array(X)

    print(" Done!", flush=True)

    # Use the classifier to predict keys for the segmented audio
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Predicting keystrokes in audio file '{audio_file}' using device {device}")

    X = torch.tensor(X.astype(np.float32))
    X = X.view(-1, 1, *X.shape[1:])

    # Batch the segmented audio for prediction
    seg_dataset = CustomDataset(X)
    seg_dataloader = DataLoader(seg_dataset, batch_size=16, shuffle=False)

    pred_y = np.array([])
    for data in seg_dataloader:
        predictions = classifier(data).cpu().detach().numpy()

        predictions = filter_by_confidence(predictions, 0.9)

        if predictions:
            transformed_pred = transformers['encoder'].inverse_transform(predictions).squeeze()
            pred_y = np.concat((pred_y, filter_predictions(transformed_pred)), axis=0)

    print("Predicted:\n\t", end="")
    emulate_typing(pred_y)


def main():
    analyze_audio('data/data1.wav')

if __name__ == '__main__':
    main()