"""
A diffusion model is designed and implemented on the MNIST dataset.
Maybe later, I'll design and train a classifier to generate class conditional samples
using a gradient-based Markov Chain Monte Carlo method..
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, utils

import numpy as np
import matplotlib.pyplot as plt
import math
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device:{device}")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
mnist_train = torchvision.datasets.MNIST(root='./data', train = True, download = True,
                                         transform = transform)
batch_size = 128
train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle = True)


# Diffusion noise schedule
class DiffusionNoiseSchedule:
  """
  Implements the forward (noising) process for a DDPM-style diffusion model
  """
  def __init__(self,timesteps=1000,beta_start=1e-4,beta_end=0.02, device=device):
    self.timesteps = timesteps
    self.device = device

    self.betas = torch.linspace(beta_start,beta_end,timesteps).to(device)

    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas,dim=0)
    self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],(1,0),value=1.0)

    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_complement_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    # denoising helpers
    self.sqrt_recip_alphas = torch.sqrt(1.0/self.alphas)

    # posterior variance
    self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev)/(1.0 - self.alphas_cumprod)

  def q_sample(self,x_0,t,noise=None):
    """
    q(x_t | x_0): add Gaussian noise at timestep t
    """
    if noise is None:
      noise = torch.randn_like(x_0)
    sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
    sqrt_complement_alpha_bar_t = self.sqrt_complement_cumprod[t].view(-1,1,1,1)

    return sqrt_alpha_bar_t*x_0 + sqrt_complement_alpha_bar_t*noise

  def get_index_from_list(self, vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))

noise_schedule = DiffusionNoiseSchedule(
    timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    device=device
)


# Sinusoidal time embeddings
class SinsoidalPositionEmbeddings(nn.Module):
  """
  Encodes timestep as vector using sinsoidal functions.
  """
  def __init__(self,dim):
    super().__init__()
    self.dim = dim

  def forward(self,time):
    device = time.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device = device) * -embeddings)
    embeddings = time[:,None] * embeddings[None,:]
    embeddings = torch.cat((embeddings.sin(),embeddings.cos()), dim = 1)
    return embeddings

class Block(nn.Module):
  """Basic Conv block with GroupNorm, ReLU, and time embedding."""

  def __init__(self, in_channels, out_channels, time_emb_dim):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.norm1 = nn.GroupNorm(8, out_channels)

    self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    # second conv takes out_channels as input
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    self.norm2 = nn.GroupNorm(8, out_channels)

    self.act = nn.ReLU()

    if in_channels != out_channels:
      self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    else:
      self.skip = nn.Identity()

  def forward(self, x, time_emb):
    # First conv
    h = self.conv1(x)
    h = self.norm1(h)
    h = self.act(h)

    # Time embedding
    time_emb = self.act(self.time_mlp(time_emb))   
    time_emb = time_emb[:, :, None, None]           
    h = h + time_emb                                 

    # Second conv
    h = self.conv2(h)
    h = self.norm2(h)
    h = self.act(h)

    # Residual
    return h + self.skip(x)

class UNet(nn.Module):
  """
  U-Net for MNIST diffusion model.

  Architecture:
  -Encoder: 28x28 -> 14x14 -> 7x7
  -bottleneck: process at 7x7
  -decoder: 7x7 -> 14x14 -> 28x28

  input: noisy image x_t + time step t
  output: predicted noise
  """

  def __init__(self, img_channels=1,time_emb_dim=128,base_channels=64):
    super().__init__()

    self.time_mlp = nn.Sequential(
        SinsoidalPositionEmbeddings(time_emb_dim),
        nn.Linear(time_emb_dim, time_emb_dim * 4),
        nn.ReLU(),
        nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
    )

    # initial convolution
    self.conv_in = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)
    # Encoder
    self.down1 = Block(base_channels, base_channels, time_emb_dim * 4)
    self.down2 = Block(base_channels, base_channels * 2, time_emb_dim * 4)
    self.pool1 = nn.MaxPool2d(2) # 28x28 -> 14x14
    self.down3 = Block(base_channels * 2, base_channels * 2, time_emb_dim * 4)
    self.down4 = Block(base_channels * 2, base_channels * 4, time_emb_dim * 4)
    self.pool2 = nn.MaxPool2d(2)  # 14x14 -> 7x7
    # Bottleneck
    self.bottleneck1 = Block(base_channels * 4, base_channels * 4, time_emb_dim * 4)
    self.bottleneck2 = Block(base_channels * 4, base_channels * 4, time_emb_dim * 4)
    # Decoder
    self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride = 2)
    self.up_block1 = Block(base_channels * 6, base_channels *2, time_emb_dim * 4)
    self.up_block2 = Block(base_channels * 2, base_channels *2, time_emb_dim * 4)
    self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)  # 14x14 -> 28x28
    self.up_block3 = Block(base_channels * 3, base_channels, time_emb_dim * 4)
    self.up_block4 = Block(base_channels, base_channels, time_emb_dim * 4)
    # Output layer
    self.conv_out = nn.Conv2d(base_channels, img_channels, kernel_size=1)

  def forward(self, x, t):
    """

    x: noisy images
    t: timesteps

    returns predicted noise
    """

    t_emb = self.time_mlp(t)

    x1 = self.conv_in(x)
    # Encoder path 
    x2 = self.down1(x1, t_emb)
    x3 = self.down2(x2, t_emb)
    x3_pooled = self.pool1(x3)
    x4 = self.down3(x3_pooled, t_emb)
    x5 = self.down4(x4, t_emb)
    x5_pooled = self.pool2(x5)

    # Bottleneck
    x = self.bottleneck1(x5_pooled, t_emb)
    x = self.bottleneck2(x, t_emb)
    # Decoder path 
    x = self.up1(x)
    x = torch.cat([x, x5], dim=1)  # skip connection: concat encoder features
    x = self.up_block1(x, t_emb)
    x = self.up_block2(x, t_emb)
    x = self.up2(x)
    x = torch.cat([x, x3], dim=1)  # skip connection: concat encoder features
    x = self.up_block3(x, t_emb)
    x = self.up_block4(x, t_emb)
    # Output (predict noise)

    return self.conv_out(x)

model = UNet(img_channels=1, time_emb_dim=128, base_channels=64).to(device)


# Training Loop
def train_diffusion_model(model, noise_schedule, train_loader, epochs =20, lr=2e-4):

  optimizer = optim.Adam(model.parameters(), lr=lr)
  model.train()

  losses = []

  for epoch in range(epochs):
    epoch_loss = 0

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        batch_size = images.shape[0]

        t = torch.randint(0, noise_schedule.timesteps, (batch_size,), device = device)
        noise = torch.randn_like(images)
        noisy_images = noise_schedule.q_sample(images,t,noise)
        predicted_noise = model(noisy_images,t)
        loss = F.mse_loss(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
          print(f"Epoch[{epoch+1}/{epochs}] Batch[{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")

    avg_epoch_loss = epoch_loss/len(train_loader)
    losses.append(avg_epoch_loss)
    print(f"Epoch[{epoch+1}/{epochs}] Avg Loss: {avg_epoch_loss:.6f}")

  return losses

# Train the diffusion model
epochs = 20
start_time = time.time()
losses = train_diffusion_model(model, noise_schedule, train_loader, epochs=epochs, lr=2e-4)
end_time = time.time()
print(f"\nTraining completed in {(end_time-start_time)/60:.2f} minutes.")

plt.figure(figsize=(10,4))
plt.plot(losses, marker='o')
plt.title('Diffusion Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.show()

def sample_ddim(model, noise_schedule, num_samples=30, img_size=28, ddim_steps=30):
    """
    DDIM sampling on CPU to avoid CUDA OOM.
    """
    original_device = next(model.parameters()).device
    model.eval().to('cpu')

    alphas_cumprod = noise_schedule.alphas_cumprod.cpu()

    with torch.no_grad():
        x = torch.randn(num_samples, 1, img_size, img_size)   # Starting noise
        timesteps = torch.linspace(
            noise_schedule.timesteps - 1, 0,
            ddim_steps, dtype=torch.long
        )

        print(f"Generating {num_samples} samples with DDIM on CPU...")

        for i, t in enumerate(timesteps):
            t = t.long()
            t_batch = t.repeat(num_samples)

            # εθ(x_t, t)
            eps = model(x, t_batch)

            alpha_t = alphas_cumprod[t]
            alpha_prev = (
                alphas_cumprod[timesteps[i+1].long()]
                if i < ddim_steps - 1 else
                torch.tensor(1.0)
            )

            # xθ prediction
            x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            x0 = x0.clamp(-1, 1)

            # x_{t-1}
            x = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1 - alpha_prev) * eps

        samples = x.cpu()

   
    model.to(original_device).train()

    return samples

generated_ddim = sample_ddim(model, noise_schedule, num_samples=30, ddim_steps=30)
def show_generated_images(images, title="Generated Digits"):
    images = images.cpu()
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    grid = utils.make_grid(images, nrow=8, padding=2)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
show_generated_images(generated_ddim, title="Generated Digits (DDIM Sampling, CPU)")
