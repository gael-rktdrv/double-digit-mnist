import numpy as np
import torch
import torch.nn as nn
import utils_multiMNIST as U
import time
from train_utils import batch_data, run_epoch, train_model

path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 128
nb_classes = 10
n_epochs = 50
num_classes = 10
img_rows, img_cols = 42, 28  # input image dimensions

# Choosing device
device = "cuda" if torch.cuda.is_available() else "cpu"


class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.neural_net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(6*19*12, 120),
            nn.LeakyReLU(),
            nn.Dropout(p=.5),
            nn.Linear(120, 20)
        )

    def forward(self, x):
        y_ = self.neural_net(x)
        out_first_digit = y_[:, :10]
        out_second_digit = y_[:, 10:]

        return out_first_digit, out_second_digit


def main():
    print(f"Device: {device}")
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batch_data(X_train, y_train, batch_size)
    dev_batches = batch_data(X_dev, y_dev, batch_size)
    test_batches = batch_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension).to(device)

    # Train
    train_model(train_batches, dev_batches, model, n_epochs=n_epochs)

    # Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print(f'\nTest loss1: {loss[0]:.6f}  accuracy1: {acc[0]:.6f}  loss2: {loss[1]:.6f}   accuracy2: {acc[1]:.6f}')


if __name__ == '__main__':
    start = time.time()
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
    end = time.time()
    print(f'\nDuration for {n_epochs} epochs with {device}: {end - start:.4f} s.\n')
