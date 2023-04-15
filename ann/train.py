import os
import sys

import numpy as np
import pandas as pd

from preprocess import prepare_dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from model.ANN import ANN


def train(learning_rate, batch_size, num_epochs, train_set, val_set, use_ckpt=False, save_ckpt=True):
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
    # Get the first batch of data
    first_batch = next(iter(train_loader))
    input_size = first_batch[0].shape[1]
    model = ANN(input_size=input_size, hidden_size=1024, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    if use_ckpt:
        if os.path.exists('model/model_checkpoint.pth'):
            checkpoint = torch.load('model/model_checkpoint.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    val_loss = 0
    best_val_loss = sys.maxsize
    best_epoch = 0
    writer = SummaryWriter(log_dir='runs/CarPricePrediction')
    writer.add_graph(model, first_batch[0])

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            # forward
            y_ = model(x)
            loss = criterion(y_.view(-1), y)

            # backward
            optimizer.zero_grad()
            loss.backward()

            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)
            # gradient descent or adam step
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        # validate the model
        scheduler.step()
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in val_loader:
                y_ = model(x)
                loss = criterion(y_.view(-1), y)
                losses.append(loss.item())
        val_loss = sum(losses) / len(losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'model/ann_best.ckpt')
        print(f'Current validation loss: {val_loss:.4f}', f'Best validation loss: {best_val_loss:.4f}',
              f'Best epoch: {best_epoch}')
        writer.add_scalar('Loss/val', val_loss, epoch)
        # save the model
        for name, param in model.named_parameters():
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            writer.add_histogram(f'Weights/{name}', param, epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, 'model/model_checkpoint.pth')
    writer.close()
    return val_loss


def hyperparameter_tuning():
    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    batch_sizes = [32, 64, 128, 256]
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    # num_epochs = [10, 20, 30]
    train_set, val_set = prepare_dataset()
    min_val_loss = sys.maxsize
    loss_table = np.zeros((len(learning_rates), len(batch_sizes), len(dropout_rates)))
    for lr in learning_rates:
        for bs in batch_sizes:
            for dropout_rate in dropout_rates:
                # for ne in num_epochs:
                val_loss = train(lr, bs, 1, train_set, val_set, use_ckpt=False, save_ckpt=False)
                loss_table[learning_rates.index(lr), batch_sizes.index(bs), dropout_rates.index(dropout_rate)] = val_loss
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_lr = lr
                    best_bs = bs
    # draw the loss table
    print(loss_table)
    loss_table = pd.DataFrame(loss_table)
    loss_table.to_csv('loss_table.csv')
    print(f'Best learning rate: {best_lr}', f'Best batch size: {best_bs}', f'Min validation loss: {min_val_loss:.4f}')


if __name__ == '__main__':
    # hyperparameter_tuning()
    train_set, val_set = prepare_dataset()
    # train(0.0005, 32, 100, train_set, val_set, use_ckpt=True)
