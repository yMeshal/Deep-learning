"""
This code designs, implements and trains a convolutional neural network (CNN) for classification on the MNIST dataset of handwritten images.
Copied from my colab.
"""
import torch, random, numpy as np
# For reproducability
def set_seed(seed: int = 42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Loading MNIST Dataset
import torchvision
import torchvision.transforms as transforms
# Defining transformation
mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
# 0.1307 = mean of the data, and 0.3081 is std.

#Load dataset
mnist_train = torchvision.datasets.MNIST(root='./data', train=True,download=True,transform=mnist_transforms)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False,download=True,transform=mnist_transforms)
print(f'Training samples:{len(mnist_train)}')
print(f'Test samples:{len(mnist_test)}')
print(f'Image shape:{mnist_train[0][0].shape}')

# Building CNN
import torch.nn as nn
import torch.nn.functional as F
# This time I'll use classes
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, channels=[32, 64]):
        """
        input_channels: 1 for MNIST, 3 for CIFAR-10
        num_classes: 10 for both datasets
        """
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=channels[0], kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if input_channels == 1:  # MNIST: 28×28
            fc_input_size = channels[1] * 7 * 7
        else:  # CIFAR-10: 32×32
            fc_input_size = channels[1] * 8 * 8

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)


        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)


        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

from torch.utils.data import DataLoader

batch_size = 64
train_loader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(mnist_test,batch_size=batch_size,shuffle=False)
print(f'Number of training batches:{len(train_loader)}')
print(f'Number of test batches:{len(test_loader)}')

def train_one_epoch (model, train_loader,criterion,optimizer,device):

  model.train()

  running_loss = 0
  correct = 0
  total = 0

  for batch_number, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    # forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

    running_loss += loss.item()
    _,predicted = torch.max(outputs,1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  epoch_loss = running_loss/len(train_loader)
  epoch_acc = 100 * correct/total

  return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100.0 * correct / total

    return test_loss, test_acc

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    print(f"Training for {num_epochs} epochs...")
    print("="*70)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("="*70)
    print("Training complete!")

    return model, history

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Liberation Sans'
def plot_training_curves(history):

    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'royalblue', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'firebrick', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.1)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'royalblue', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'firebrick', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.1)

    plt.tight_layout()
    plt.show()

"""
I performed experiments through changing architectures (i.e., numbers of channels, kernel sizes, depth
in each layer) and training hyperparameters (i.e., batch size, learning rate, momentum, etc.) in an effort to improve the test accuracy. 

MNIST was simple enough for the CNN to achieve great accuracy. The CIFAR was proven to be much more diffcult. 
"""

import time
experiment_name = 'MNIST-Baseline'
input_channels = 1
channels = [32, 64]
batch_size = 64
learning_rate = 0.001
optimizer_type = 'adam'
num_epochs = 10

print(f"\n{'='*70}")
print(f"EXPERIMENT: {experiment_name}")
print(f"{'='*70}")
print(f"Channels: {channels}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {learning_rate}")
print(f"Optimizer: {optimizer_type}")
print(f"{'='*70}\n")

model = SimpleCNN(input_channels=input_channels, num_classes=10, channels=channels).to(device)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
print(f'Number of training batches:{len(train_loader)}')
print(f'Number of test batches:{len(test_loader)}')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params:,}\n")

criterion = nn.CrossEntropyLoss()
if optimizer_type == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

start_time = time.time()
model, history = train_model(model, train_loader, test_loader,
                                         criterion, optimizer, device, num_epochs)
training_time = time.time() - start_time

best_test_acc = max(history['test_acc'])
best_epoch = history['test_acc'].index(best_test_acc) + 1
final_train_acc = history['train_acc'][-1]

print(f"\n{'='*70}")
print(f"RESULTS: {experiment_name}")
print(f"{'='*70}")
print(f"Best Test Acc: {best_test_acc:.2f}% (epoch {best_epoch})")
print(f"Final Train Acc: {final_train_acc:.2f}%")
print(f"Training Time: {training_time:.1f}s")
print(f"Parameters: {total_params:,}")
print(f"{'='*70}\n")

plot_training_curves(history)

print(f"{experiment_name} | {channels} | {batch_size} | {learning_rate} | {optimizer_type} | {best_test_acc:.2f} | {best_epoch} | {final_train_acc:.2f} | {training_time:.1f} | {total_params}")
