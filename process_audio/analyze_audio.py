import numpy as np
import torch
from deep_learning_functions import load_model, load_transformers, transform_data, emulate_typing
from preprocess_data import load_data, unlabeled_audio_segmentation, convert_to_array

def analyze_audio(audio_file: str, models_path='model_data/'):
    print("Loading data...", end='')
    # Load model
    classifier = load_model('classifier', path=models_path)

    # Load transformers
    transformers = load_transformers(models_path + '/classifier/transformer_dumps/')

    # Load audio
    audio, _, sr = load_data(label_file=None, audio_file=audio_file)

    print(' Done!')

    # Segment audio using the detector model
    X = unlabeled_audio_segmentation(audio, models_path)

    # Modify and transform the segmented audio
    X = convert_to_array(X)
    X, _ = transform_data(X, None, transformers)

    # Use the classifier to predict keys for the segmented audio
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Predicting keystrokes in audio file '{audio_file}' using device {device}")

    predictions = classifier(torch.tensor(X.astype(np.float32)).to(device)).cpu().detach().numpy()
    pred_y = transformers['encoder'].inverse_transform(predictions).squeeze()

    print("Predicted:\n\t", end="")
    emulate_typing(pred_y)
    



def main():
    analyze_audio('data/data1.wav')

if __name__ == '__main__':
    main()