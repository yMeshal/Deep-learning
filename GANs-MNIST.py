"""
A generative adversarial network is designed and implemented on the MNIST dataset of handwritten images. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
batch_size = 128
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

print(f'Training samples: {len(mnist_train)}')
print(f'Batches: {len(train_loader)}')


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 128 * 7 * 7)
        self.bn1 = nn.BatchNorm1d(128 * 7 * 7)

        # 7×7 → 14×14
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 14×14 → 28×28
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc1(z)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.view(x.size(0), 128, 7, 7)

        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.deconv2(x)
        x = torch.tanh(x)
        return x


# DISCRIMINATOR 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256 * 3 * 3, 1)

    def forward(self, img):
        x = self.conv1(img)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss and optimizerss
criterion = nn.BCEWithLogitsLoss()
lr = 2e-4
beta1 = 0.5

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))


def create_labels(size, value, device):
    return torch.full((size, 1), value, dtype=torch.float, device=device)


# TRAINING LOOP
num_epochs = 500
G_losses = []
D_losses = []

for epoch in range(num_epochs):
    epoch_g_losses = []
    epoch_d_losses = []

    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        real_labels = create_labels(batch_size, 0.9, device)
        fake_labels = create_labels(batch_size, 0.1, device)

        # Train Discriminator 
        optimizer_D.zero_grad()

        # Real images
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)

        # Fake images
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator 2x (twice as much)
        for _ in range(2):
            optimizer_G.zero_grad()

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images)

            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

        epoch_g_losses.append(g_loss.item())
        epoch_d_losses.append(d_loss.item())

    G_losses.append(np.mean(epoch_g_losses))
    D_losses.append(np.mean(epoch_d_losses))

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {D_losses[-1]:.4f} | G Loss: {G_losses[-1]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(G_losses, label='Generator Loss', linewidth=2)
plt.plot(D_losses, label='Discriminator Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Losses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

generator.eval()
with torch.no_grad():
    noise = torch.randn(64, latent_dim, device=device)
    fake_images = generator(noise).cpu()

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
fig.suptitle('Generated MNIST Digits', fontsize=16, fontweight='bold')

for i in range(64):
    row = i // 8
    col = i % 8
    img = fake_images[i].squeeze()
    img = (img + 1) / 2
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
