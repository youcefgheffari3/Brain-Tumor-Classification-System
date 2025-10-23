# gan_brain_fixed.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# ---- CONFIG ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
batch_size = 128
z_dim = 256
image_size = 64
image_channels = 1
image_dim = image_size * image_size * image_channels
lr = 3e-4
num_epochs = 100
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# ---- TRANSFORMS ----
# For GAN with Tanh output, normalize images to [-1, 1]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize((0.5,), (0.5,)),  # -> [-1,1]
])

# ---- SIMPLE CustomImageDataset (use your implementation if you already have one) ----
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # expects folder with subfolders per class or images directly
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # return image normalized (transform already applied) and label
        return img, label

# Replace with your actual dataset root
root_dir = '/home/dairi/Datasets/Brain Tumor/imgResized/'
full_dataset = CustomImageDataset(root_dir=root_dir, transform=transform)

n = len(full_dataset)
n_train = int(0.8 * n)
n_test = n - n_train
from torch.utils.data import random_split
train_set, test_set = random_split(full_dataset, [n_train, n_test])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

# ---- MODELS (simple MLP GAN as original) ----
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim, hidden_dim=1024):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, img_dim),
            nn.Tanh(),  # outputs in [-1,1]
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim, hidden_dim=1024):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

generator = Generator(z_dim, image_dim).to(device)
discriminator = Discriminator(image_dim).to(device)

optG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# For TensorBoard
writer_fake = SummaryWriter("logs/fake")
writer_real = SummaryWriter("logs/real")

fixed_noise = torch.randn((batch_size, z_dim), device=device)

# ---- TRAIN ----
step = 0
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(train_loader):
        if real.size(0) != batch_size:
            continue
        real = real.view(batch_size, -1).to(device)

        # Train Discriminator
        optD.zero_grad()
        noise = torch.randn(batch_size, z_dim, device=device)
        fake = generator(noise)
        d_real = discriminator(real).view(-1)
        d_fake = discriminator(fake.detach()).view(-1)
        lossD_real = criterion(d_real, torch.ones_like(d_real))
        lossD_fake = criterion(d_fake, torch.zeros_like(d_fake))
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward()
        optD.step()

        # Train Generator
        optG.zero_grad()
        output = discriminator(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        lossG.backward()
        optG.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} LossD: {lossD.item():.4f} LossG: {lossG.item():.4f}")

            # visualize
            with torch.no_grad():
                fake_imgs = generator(fixed_noise).reshape(-1, 1, image_size, image_size)
                real_imgs = real.view(-1, 1, image_size, image_size)
                img_grid_fake = torchvision.utils.make_grid(fake_imgs, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_imgs[:batch_size], normalize=True)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                step += 1

# ---- Save models ----
torch.save(generator.state_dict(), os.path.join(model_dir, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(model_dir, "discriminator.pth"))
print("Training finished, models saved.")
