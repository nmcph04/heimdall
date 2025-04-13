import numpy as np
import torch
from deep_learning_functions import load_model, load_transformers, transform_data, emulate_typing
from preprocess_data import load_data, unlabeled_audio_segmentation, convert_to_array

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

# Filters predictions so that one character isn't predicted more than twice in a row
def filter_predictions(pred: str) -> list:
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
    classifier = load_model('classifier', path=models_path)

    # Load transformers
    transformers = load_transformers(models_path + '/classifier/transformer_dumps/')

    # Load audio
    audio, _, sr = load_data(label_file=None, audio_file=audio_file)

    audio = audio[:5000000]

    print(' Done!', flush=True)

    # Segment audio using naive segmentation
    print("Segmenting audio...", end='', flush=True)
    X = naive_segmentation(audio, sr=sr)

    # Modify and transform the segmented audio
    X = convert_to_array(X)
    X, _ = transform_data(X, None, transformers)

    print(" Done!", flush=True)

    # Use the classifier to predict keys for the segmented audio
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Predicting keystrokes in audio file '{audio_file}' using device {device}")

    predictions = classifier(torch.tensor(X.astype(np.float32)).to(device)).cpu().detach().numpy()

    predictions = filter_by_confidence(predictions, 0.9)

    pred_y = transformers['encoder'].inverse_transform(predictions).squeeze()
    pred_y = filter_predictions(pred_y)

    print("Predicted:\n\t", end="")
    emulate_typing(pred_y)
    print(len(pred_y))


def main():
    analyze_audio('data/data1.wav')

if __name__ == '__main__':
    main()