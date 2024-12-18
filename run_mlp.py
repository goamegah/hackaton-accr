import torch
import argparse
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn

from accr.utils.evaluate import set_all_seeds
from accr.utils.data import train_test_split_custom
from accr.data.processing import Processing

from accr.models.mlp import MLP

def normalize_data(X_train, X_test):
    X_train_normalized = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test_normalized = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    return X_train_normalized, X_test_normalized

parser = argparse.ArgumentParser(description='Custom MLP')

parser.add_argument('-j', '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--disable-cuda',
                    action='store_true',
                    help='Disable CUDA')

parser.add_argument('--seed',
                    default=1,
                    type=int,
                    help='seed for initializing training.')

parser.add_argument('--alpha',
                    default=0.01,
                    type=float,
                    help='Learning rate for MLP training.')

parser.add_argument('--epochs',
                    default=1000,
                    type=int,
                    help='Number of training epochs.')

def main():
    args = parser.parse_args()
    processing = Processing()

    # Configurer le dispositif GPU/CPU
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    set_all_seeds(args.seed)

    # Charger et préparer les données
    train_df = pd.read_csv('./dataset/preprocessed_data.csv')
    train_ds = train_df.apply(pd.to_numeric, errors='coerce')

    y = train_ds['grav'] - 1  # Ajuster les labels pour commencer à 0
    X = train_ds.drop(columns=['grav'])

    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalize_data(X_train.values, X_test.values)

    # Conversion en tableaux compatibles avec le MLP
    y_train_one_hot = np.eye(4)[y_train.values.astype(int)]  # Sortie au format one-hot
    y_test_one_hot = np.eye(4)[y_test.values.astype(int)]

    # Initialisation du modèle MLP
    input_dim = X_train.shape[1]
    hidden_dim = 32
    output_dim = 4
    model = MLP([input_dim, hidden_dim, output_dim])

    # Entraînement du modèle
    model.train(
        all_samples_inputs=np.array(X_train),
        all_samples_expected_outputs=np.array(y_train_one_hot),
        alpha=args.alpha,
        nb_iter=args.epochs,
        is_classification=True
    )


    # Évaluation
    correct = 0
    for i in range(len(X_test)):
        prediction = model.predict(X_test[i], is_classification=True)
        predicted_label = np.argmax(prediction)
        if predicted_label == y_test.values[i]:
            correct += 1

    accuracy = correct / len(X_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
