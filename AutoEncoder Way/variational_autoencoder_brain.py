# variational_autoencoder_brain_fixed.py
# -*- coding: utf-8 -*-
import os
import pickle
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---- CONFIG ----
img_dir = '/home/dairi/Datasets/Brain Tumor/imgResized/'  # update if needed
batch_size = 125
image_size = 64  # assuming images are 64x64
input_dim = image_size * image_size  # 4096
latent_dim = 256
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# ---- TRANSFORMS ----
# VAE uses Sigmoid on output => inputs in [0,1]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),  # yields [0,1]
])

# ---- DATASET & DATALOADERS ----
dataset = datasets.ImageFolder(img_dir, transform=transform)
print("Classes:", dataset.classes)
n = len(dataset)
n_train = int(0.8 * n)
n_test = n - n_train
train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

print(f"Dataset size: {n}  Train: {n_train}  Test: {n_test}")
print('Sample image shape:', next(iter(train_loader))[0][0].shape)

# ---- MODEL ----
class VAE(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024, latent_dim=256):
        super().__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        # latent parameters
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # because inputs in [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(mean.device)
        return mean + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

# instantiate
model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

# loss: BCE reconstruction + KL divergence
def loss_function(x, x_hat, mean, logvar):
    # reconstruction loss: sum over pixels, average later over batch
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kld

# training loop
def train(model, optimizer, epochs=50):
    model.train()
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(x.size(0), -1).to(device)
            optimizer.zero_grad()
            x_hat, mean, logvar = model(x)
            loss = loss_function(x, x_hat, mean, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_dataset)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} avg_loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(model_dir, 'vae_latest.pth'))
        with open(os.path.join(model_dir, 'vae_loss_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
    return history

# ---- Run or load ----
do_train = False
if do_train:
    history = train(model, optimizer, epochs=100)
else:
    ckpt = os.path.join(model_dir, 'vae_latest.pth')
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("Loaded checkpoint:", ckpt)
    else:
        print("No checkpoint found, starting with random weights.")

# ---- Utility: generate and plot ----
def generate_and_plot(model, z_vals, image_size=64):
    model.eval()
    with torch.no_grad():
        z = torch.tensor(z_vals, dtype=torch.float32).to(device)
        x_decoded = model.decode(z).cpu().numpy()
    for i, img_flat in enumerate(x_decoded):
        img = img_flat.reshape(image_size, image_size)
        plt.figure()
        plt.title(f"z={z_vals[i].tolist()}")
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

# Example latent grid visualization (if latent_dim >= 2)
if latent_dim >= 2:
    # sample a small grid in first two dims, zeros for rest
    n = 4
    scale = 1.0
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    z_list = []
    for yi in grid_y:
        for xi in grid_x:
            z = np.zeros(latent_dim, dtype=np.float32)
            z[0] = xi
            z[1] = yi
            z_list.append(z)
    generate_and_plot(model, z_list[:16], image_size=image_size)

# ---- Encode test set and save representation ----
model.eval()
encoded_list = []
label_list = []
with torch.no_grad():
    for imgs, labels in test_loader:
        batch = imgs.view(imgs.size(0), -1).to(device)
        # encoder returns latent vectors prior to mean/logvar layer in our architecture, but we want mean
        mean, logvar = model.encode(batch)
        encoded_list.append(mean.cpu().numpy())
        label_list.append(labels.numpy())
encoded = np.vstack(encoded_list)
labels = np.concatenate(label_list)
print("Encoded shape:", encoded.shape, "Labels shape:", labels.shape)
with open(os.path.join(model_dir, 'test_encoded_dataset.pkl'), 'wb') as f:
    pickle.dump([encoded, labels], f)
print("Saved encoded dataset.")
