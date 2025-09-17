# Python imports
import os
from tqdm import tqdm

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Helper functions for computer vision
import torchvision
import torchvision.transforms as transforms

# Math imports
import math

# Create an empty 'results.txt' file
with open('results.txt', 'w'):
    pass

class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.flatten = nn.Flatten()

        # Layer 4
        self.fc1 = nn.Linear(16 * 5 * 5, 256, bias=True)
        self.relu3 = nn.ReLU()

        # Layer 5
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.relu4 = nn.ReLU()

        # Layer 6
        self.fc3 = nn.Linear(128, num_classes, bias=True)

    def forward(self, x):
        shape_dict = {}

        # Layer 1 operations
        x = self.maxpool1(self.relu1(self.conv1(x)))
        shape_dict[1] = x.shape[1:]

        # Layer 2 operations
        x = self.maxpool2(self.relu2(self.conv2(x)))
        shape_dict[2] = x.shape[1:]

        # Layer 3 operations
        x = self.flatten(x)
        shape_dict[3] = x.shape[1:]

        # Layer 4 operations
        x = self.relu3(self.fc1(x))
        shape_dict[4] = x.shape[1:]

        # Layer 5 operations
        x = self.relu4(self.fc2(x))
        shape_dict[5] = x.shape[1:]

        # Layer 6 operations
        out = self.fc3(x)
        shape_dict[6] = out.shape[1:]

        return out, shape_dict

def count_model_params():
    '''
    Return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Convert to millions
    model_params_in_millions = model_params / 1e6

    return model_params_in_millions

def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0.0

    for input, target in tqdm(train_loader, total=len(train_loader)):
        # 1) Zero the parameter gradients
        optimizer.zero_grad()
        # 2) Forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss

def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0

    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(epoch+1, 100. * test_acc))

    return test_acc
