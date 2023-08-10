import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from nn_mnistModel import Model
import json
import os
import sys
from lib.mypckg import dataloader

epochs = 10
batch_size = 128
dst_directory = 'C:/Users/Admin/Desktop/VS/project/main/out/nn_mnist'
weight_path = os.path.join(os.path.normpath(dst_directory),'nn_mnistModel.pth')
history_path = os.path.join(os.path.normpath(dst_directory),'nn_mnistModel_history.json')
fn_data = 'C:/Users/Admin/Desktop/VS/project/main/data/mnist_data.csv'
#Load dataset
x_data, y_data = dataloader.load_data(filename=fn_data, res_cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                              drop_na = False, col_names = False, 
                              inp_shape = (1, 1, 28, 28), out_shape = (1, 10))


train_size = int(len(x_data) * 80.0 * 0.01)
val_size = int((len(x_data) - train_size) / 2)
test_size = len(x_data) - train_size - val_size

# Создание TensorDataset из тензоров
dataset = TensorDataset(torch.from_numpy(x_data), torch.from_numpy(y_data))
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

# Создание DataLoader
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = Model()

#Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters()) # lr=1e-3

def train(dataloader, model, loss_fn, optimizer, metric):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, train_metric = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_metric += metric_fn(pred, y,metric)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
    train_loss /= num_batches
    train_metric /= size
    print(f"loss: {train_loss:>8f}, {metric}: {train_metric:>8f}", flush=True)
    return train_loss, train_metric

def test(dataloader, model, loss_fn, metric, lb = 'test'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_metric = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_metric += metric_fn(pred, y, metric) 
    test_loss /= num_batches
    test_metric /= size
    print(f"{lb}_loss: {test_loss:>8f}, {lb}_{metric}: {test_metric:>8f} \n", flush=True)
    return test_loss, test_metric

def metric_fn(pred, y, metric = 'mae'):
    if metric == 'acc':
        return (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    if metric == 'mse':
        loss_fn = nn.MSELoss()
    if metric == 'mae':
        loss_fn = nn.L1Loss(reduction = 'sum')
    return loss_fn(pred, y).item()

keys = ['loss', 'acc', 'val_loss', 'val_acc']
history_dict = {key: [] for key in keys}

# Train the model
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------", flush=True)
    loss, metric = train(train_dataloader, model, loss_fn, optimizer,'acc')
    val_loss, val_metric = test(val_dataloader, model, loss_fn, 'acc', 'val')

    metrics = {'loss': loss, 'acc': metric, 'val_loss': val_loss, 'val_acc': val_metric}
    for key, value in metrics.items():
        history_dict[key].append(value)

# Save history to json file
with open(history_path, 'w') as f:
    json.dump(history_dict, f)

# Evaluate the model on the test data 
print("Evaluate on test data", flush=True)
test(test_dataloader, model, loss_fn, 'acc')

print("Generate predictions for 3 samples", flush=True)
with torch.no_grad():
    inputs = train_data[:3][0]
    predictions = model(inputs)
    print("predictions shape:", predictions.shape, flush=True)

print('PyTorch model saved in: ', weight_path, flush=True)
torch.save(model.state_dict(), weight_path)