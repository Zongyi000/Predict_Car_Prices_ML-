import torch
# import jovian
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


DATA_FILENAME = "./linear_regressing_car_data.csv"
dataframe_raw = pd.read_csv(DATA_FILENAME)
dataframe_raw.head()


your_name = "Zongyi Jiang"
def customize_dataset(dataframe_raw, rand_str):
    dataframe = dataframe_raw.copy(deep=True)
    # drop some rows
    dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))
    # scale input
    dataframe.Year = dataframe.Year * ord(rand_str[1])/100.
    # scale target
    dataframe.Selling_Price = dataframe.Selling_Price * ord(rand_str[2])/100.
    # drop column
    if ord(rand_str[3]) % 2 == 1:
        dataframe = dataframe.drop(['Car_Name'], axis=1)
    return dataframe

dataframe = customize_dataset(dataframe_raw, your_name)
dataframe.head()

# print(dataframe)

input_cols = ["Year","Present_Price","Kms_Driven","Owner"]
categorical_cols = ["Fuel_Type","Seller_Type","Transmission"]
output_cols = ["Selling_Price"]


#Data Preparation: to use the data for training we need to convert it from dataframe to PyTorch Tensors, the first step is to convert to NumPy arrays

def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array

inputs_array, targets_array = dataframe_to_arrays(dataframe)

inputs = torch.Tensor(inputs_array)
targets = torch.Tensor(targets_array)

dataset = TensorDataset(inputs, targets)
# 随机将一个数据集分割成给定长度的不重叠的新数据集 228 57为两个长度
train_ds, val_ds = random_split(dataset, [228, 57])
batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

#Creating PyTorch Model: a linear regressing model using PyTorch to predict car prices

input_size = len(input_cols)
output_size = len(output_cols)

class CarsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)    # fill this (hint: use input_size & output_size defined above)

    def forward(self, xb):
        out = self.linear(xb)                          # fill this
        return out

    def training_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calcuate loss
        loss = F.l1_loss(out, targets)                         # fill this
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out, targets)                           # fill this
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))

model = CarsModel()

list(model.parameters())


#Training Model to Predict Car Prices:assess the loss and see how much is, and after doing the training, see how much the loss decreases with training

# Eval algorithm
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Fitting algorithm
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history

# Check the initial value that val_loss have
result = evaluate(model, val_loader)
print("initial value that val_loss have:", result)




# Start with the Fitting
epochs = 300   #modified from 90 to make sure val_loss is below 10
lr = 1e-8
history1 = fit(epochs, lr, model, train_loader, val_loader)


# Train repeatdly until have a 'good' val_loss
epochs = 20
lr = 1e-9
history1 = fit(epochs, lr, model, train_loader, val_loader)


#Using the Model to Predict Car Prices:
# Prediction Algorithm
def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)                # fill this
    prediction = predictions[0].detach()
    print("Input:", input.numpy())
    print("Target:", target.numpy())
    print("Prediction:", prediction.numpy())

# Testing the model with some samples
print("------------------------")
print("Testing the model with some samples below:")

print("sample1:")
input, target = val_ds[0]
predict_single(input, target, model)
print("------------------------")

print("sample2:")
input, target = val_ds[10]
predict_single(input, target, model)
print("------------------------")



