# For image transforms
import torchvision.transforms as transforms
# For Pytorch methods
import torch
import torch.nn as nn
# For Optimizer
import torch.optim as optim
import torchvision
# For DATA SET
import torchvision.datasets as datasets
# FOR DATA LOADER
from torch.utils.data import DataLoader
# FOR TENSOR BOARD VISUALIZATION
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard

from CustomDataset import CustomImageDataset

batch_size=128
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    zDim = 256 # 64, 128, 256
    imageDim = 64 * 64 * 1  # 784
    batchSize = batch_size  # Batch size
    numEpochs = 100  # Change as per your need
    logStep = 625  # Change as per your need

#########################################################################

from torch.utils.data import DataLoader
_train_dataset = CustomImageDataset(dst_type='train')
_test_dataset = CustomImageDataset(dst_type='test')
train_dataloader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(_test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
#########################################################################
class Generator(nn.Module):
    def __init__(self, zDim, imgDim, hiddenDim=512, lr=0.01):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(zDim, hiddenDim),
            nn.LeakyReLU(lr),
            nn.Linear(hiddenDim, imgDim),
            nn.Tanh(),  # We normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)
#########################################################################
class Discriminator(nn.Module):
    def __init__(self, inFeatures, hiddenDim=512, lr=0.01):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(inFeatures, hiddenDim),
            nn.LeakyReLU(lr),
            nn.Linear(hiddenDim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

discriminator = Discriminator(Config.imageDim).to(Config.device)
generator = Generator(Config.zDim,
                    Config.imageDim).to(Config.device)

# Fixed Noise
fixedNoise = torch.randn((Config.batchSize,
                              Config.zDim)).to(Config.device)

print(f"\nSetting Optimizers")
optDisc = optim.Adam(discriminator.parameters(),
                     lr=Config.lr)
optGen = optim.Adam(generator.parameters(),
                    lr=Config.lr)
criterion = nn.BCELoss()

writerFake = SummaryWriter(f"logs/fake")
writerReal = SummaryWriter(f"logs/real")

#########################################################################
def prepareVisualization(epoch,
                         batchIdx,
                         loaderLen,
                         lossD,
                         lossG,
                         writerFake,
                         writerReal,
                         step):
    print(
        f"Epoch [{epoch}/{Config.numEpochs}] Batch {batchIdx}/{loaderLen} \
                              Loss DISC: {lossD:.4f}, loss GEN: {lossG:.4f}"
    )

    with torch.no_grad():
        # Generate noise via Generator
        fake = generator(fixedNoise).reshape(-1, 1, 64, 64)

        # Get real data
        data = real.reshape(-1, 1, 64, 64)

        # Plot the grid
        imgGridFake = torchvision.utils.make_grid(fake,
                                                  normalize=True)
        imgGridReal = torchvision.utils.make_grid(data,
                                                  normalize=True)

        writerFake.add_image("Mnist Fake Images",
                             imgGridFake,
                             global_step=step)
        writerReal.add_image("Mnist Real Images",
                             imgGridReal,
                             global_step=step)
        # increment step
        step += 1

    return step
#########################################################################
#########################   TRAINING           ##########################
#########################################################################
step = 0
doTrain=1
if doTrain :
    print(f"\nStarted Training and visualization...")
    for epoch in range(Config.numEpochs):
        print('-' * 80)
        for batch_idx, (real, _) in enumerate(train_dataloader):
            if real.shape[0] != batch_size: break
            real = real.view(batch_size,64*64).to(Config.device)
            batchSize = real.shape[0]
            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(batchSize,
                                Config.zDim).to(Config.device)
            fake = generator(noise)
            discReal = discriminator(real).view(-1)
            lossDreal = criterion(discReal,
                                  torch.ones_like(discReal))
            discFake = discriminator(fake).view(-1)
            lossDfake = criterion(discFake,
                                  torch.zeros_like(discFake))
            lossD = (lossDreal + lossDfake) / 2
            discriminator.zero_grad()
            lossD.backward(retain_graph=True)
            optDisc.step()

            ###
            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients. Minimizing is easier
            ###
            output = discriminator(fake).view(-1)
            lossG = criterion(output,
                              torch.ones_like(output))
            generator.zero_grad()
            lossG.backward()
            optGen.step()

            # Visualize three steps for each epoch
            if batch_idx % Config.logStep == 0:
                step = prepareVisualization(epoch,
                                            batch_idx,
                                            len(train_dataloader),
                                            lossD,
                                            lossG,
                                            writerFake,
                                            writerReal,
                                            step)

torch.save(discriminator.state_dict(), 'models/discriminator.pt')
torch.save(generator.state_dict(), 'models/generator.pt')

discriminator.load_state_dict(torch.load('models/discriminator.pt', weights_only=True))
generator.load_state_dict(torch.load('models/generator.pt', weights_only=True))

