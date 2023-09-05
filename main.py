from os import system
import pandas as pd
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from data import CustomDataset
from nn import NeuralNetwork


def clear():
    system('clear')

# get numpy array from csv file
df = pd.read_csv('./eurusd-5m.csv', sep=';')
closes = df['close'].to_numpy(dtype=np.float32)#[-10000:]

# data normalization
mean = closes.mean()
std = closes.std()
normalized_data = (closes - mean) / std


# parameters for data and model
chunk_size = 100
input_size = chunk_size
hidden_size = 128
output_size = 1
batch_size = 1



# dataset
dataset = CustomDataset(normalized_data, chunk_size, output_size)

# train/test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# number of chunks for each dataset
train_chunks = len(train_dataset) - (chunk_size + output_size + 1)
test_chunks = len(test_dataset) - (chunk_size + output_size + 1)

# train data loader
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# model
model = NeuralNetwork(input_size, hidden_size, output_size)


# loss function
loss_function = nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


counter = 0
for inputs, targets in train_data_loader:
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    counter += 1

    if counter % 200 == 0:
        clear()
        print(f"progress: {((counter/train_chunks)*100):.0f}%")
        print(f"loss: {loss.item():.7f}")



test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model.eval()
correct_predictions = 0
incorrect_predictions = 0

test_precision = 0

with torch.no_grad():
    counter = 0
    for inputs, targets in test_data_loader:
        targets = targets.view(-1, 1)

        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        
        original_inputs = (inputs * std) + mean
        original_targets = (targets * std) + mean
        original_outputs = (outputs * std) + mean

        test_precision += abs(original_outputs[-1][-1] - original_targets[-1][-1])

        if float(original_targets[-1][-1]) > float(original_inputs[-1][-1]) and float(original_outputs[-1][-1]) > float(original_inputs[-1][-1]):
            direction = "correct"
            correct_predictions += 1
        else:
            direction = "incorrect"
            incorrect_predictions += 1

        '''
        print("------------------------------------------------")
        print(f"price: {original_inputs[-1][-1]}")
        print(f"prediction: {original_outputs[-1][-1]}")
        print(f"actual future price: {original_targets[-1][-1]}")
        print(f"loss: {loss.item():.7f}")
        print(f"difference: {abs(float(original_outputs[-1][-1]) - float(original_targets[-1][-1]))}")
        print(f"direction: {direction}")
        '''
        counter += 1
        if counter % 200 == 0:
            clear()
            print(f"progress: {((counter/test_chunks)*100):.0f}%")
            print(f"loss: {loss.item():.7f}")


test_precision /= len(test_data_loader)
clear()
print(f"precision: {(test_precision*10000):.1f} pips")
print(f"success rate: {(correct_predictions/(correct_predictions + incorrect_predictions))*100}%")



