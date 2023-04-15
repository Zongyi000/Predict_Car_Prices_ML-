import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import joblib
from model.ANN import ANN
from preprocess import prepare_test_dataset


def test():
    test_set = prepare_test_dataset()
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    first_batch = next(iter(test_loader))
    input_size = first_batch[0].shape[1]
    print(f'Input size: {input_size}')
    model = ANN(input_size=input_size, hidden_size=1024, output_size=1)
    model.load_state_dict(torch.load(f'model/ann_best.ckpt'))
    model.eval()
    price_scaler = joblib.load('price_scaler.joblib')
    accuracy = 0
    r_squared = 0
    with torch.no_grad():
        num_samples = 0
        sum_squared_errors = 0
        for x, y in test_loader:
            y_ = model(x)
            y_ = price_scaler.inverse_transform(y_.view(-1, 1).numpy())
            y = y.view(-1, 1).numpy()
            squared_errors = (y_ - y) ** 2  # Calculate squared errors for each sample
            sum_squared_errors += squared_errors.sum()  # Sum the squared errors for the batch
            num_samples += len(y)  # Update the total number of samplesError
            abs_error = np.abs(y_ - y)
            accuracy += (abs_error / y) * 100
            print(f'Predicted: {y_[0][0]:.2f}, Actual: {y[0][0]:.2f}', f'Error: {abs_error[0][0]:.2f}')
        accuracy = accuracy / num_samples
        mse = sum_squared_errors / num_samples  # Calculate the mean squared error
        rmse = np.sqrt(mse)
        print(f'Test loss: {rmse:.4f}')


if __name__ == '__main__':
    test()
