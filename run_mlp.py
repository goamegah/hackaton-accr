import torch
import argparse
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn

from accr.utils.evaluate import set_all_seeds
from accr.models.mlp import MLP
from accr.utils.data import train_test_split_custom
from accr.data.processing import Processing
from accr.utils.early_stopping import EarlyStopping
from accr.utils.trainer import Trainer

parser = argparse.ArgumentParser(description='PyTorch MLP')

parser.add_argument('-m', '--model',
                    metavar='model',
                    default='mlp',
                    choices=["mlp"],
                    help='model to run (default: mlp)')

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

parser.add_argument('--early-stopping',
                    action='store_true',
                    help='Enable early stopping')


def main():
    args = parser.parse_args()
    processing = Processing()

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    set_all_seeds(args.seed)

    train_df = pd.read_csv('./dataset/preprocessed_data.csv')

    train_ds = train_df.apply(pd.to_numeric, errors='coerce')

    y = train_ds['grav']
    y = y - 1
    X = train_ds.drop(columns=['grav'])

    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

    X_train = (X_train - np.mean(X_train)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_train)) / np.std(X_train, axis=0)

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    if args.model == 'mlp':
        model = MLP(input_dim=X_train.shape[1], hidden_dim=32, output_dim=4).to(args.device)

    # early_stopping = EarlyStopping(patience=5, min_delta=0.01) if args.early_stopping else None

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        # early_stopping=early_stopping
    )

    trainer.fit(num_epochs=20, checkpoint_path='./output-model/best_model.pth')


if __name__ == '__main__':
    main()
