import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy, AUROC
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassConfusionMatrix, F1Score
import os
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

##############################################################################
####################DEFINE THE CNN MODEL        ##############################
##############################################################################
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes=2):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))            # Apply fully connected layer
        x = self.fc2(x)   # Apply fully connected layer
        x = F.softmax(x, dim=1)
        return x
############################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print("########### using device : ",device)
############################################################################
##############      Prepare Dataset Using Dataloaders       ################
############################################################################
num_classes = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 10  # Reduced for demonstration purposes
img_dir='/home/dairi/Datasets/Brain Tumor/imgResized/'
transform = transforms.Compose([ transforms.Grayscale(),
                                transforms.ToTensor()
                               ])
# Define transform
dataset = datasets.ImageFolder(img_dir, transform=transform)
print("Classes : ",dataset.classes)
print("Classes : ",dataset.class_to_idx)
#Split Dataset : 80% Training 20% Testing
train_dataset, test_dataset = random_split(dataset, [0.8,0.2])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,drop_last=True)

model = CNN(in_channels=1, num_classes=num_classes).to(device)
lossfnc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#######################################################################
##########################   TRAINING  ################################
#######################################################################
PATH = 'models/brain_tumor_cnn-'+str(num_epochs)+'.pth'
doTrain=False
train_loss_hist=[]
if doTrain:
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_dataloader)):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute the model output
            scores = model(data)
            loss = lossfnc(scores, targets)
            running_loss += loss.item()
            if batch_index % 10 == 9:
                train_loss_hist.append(running_loss / 10.)
                print(f'[{epoch + 1}, {batch_index + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()
    #End of Training Loop Save the model
    torch.save(model.state_dict(), PATH)
    plt.plot(train_loss_hist, label="train")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.show()

model.load_state_dict(torch.load(PATH, weights_only=True))

#######################################################################
########################## EVALUATION  ################################
#######################################################################
def evaluation(loader, model,_batch_size,name='train'):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode
    all_labels=[]
    all_preds=[]
    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            all_labels.append(y.cpu().numpy())

            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            all_preds.append(predictions.cpu().numpy())
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
        pyt_acc = Accuracy(task="multiclass", num_classes=2).to(device)
        acc = pyt_acc(torch.from_numpy(all_preds), torch.from_numpy(all_labels))
        print(f"Accuracy on {name} set: {acc*100:.5f}%")
        pyt_f1 = F1Score(task="multiclass", num_classes=2).to(device)
        f1 = pyt_f1(torch.from_numpy(all_preds), torch.from_numpy(all_labels))
        print(f"F1-Score on {name} set: {f1*100:.5f}%")
        auroc = AUROC(task="binary").to(device)
        auc = auroc(torch.from_numpy(all_preds), torch.from_numpy(all_labels))

        print(f"auroc-Score on {name} set: {auc * 100:.5f}%")

        metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        metric.update(torch.from_numpy(all_preds).to(device), torch.from_numpy(all_labels).to(device))
        fig_, ax_ = metric.plot()
        plt.show()
    model.train()  # Set the model back to training mode
    return all_preds,all_labels


# Final accuracy check on training and test sets
_,_ = evaluation(train_dataloader, model,batch_size)
_,_ = evaluation(test_dataloader, model,batch_size,name='test')


#img = torchvision.io.read_image('/home/dairi/Datasets/Brain Tumor/imgResized/tumor/171.png').to(device)
img = torchvision.io.read_image('/home/dairi/Desktop/imgs/brain4.jpg').to(device)
transform =transforms.Compose([
    transforms.ToPILImage(),transforms.Grayscale(),
    transforms.ToTensor()
])
img= transform(img).to(device)
pred = model(img.view(1, 1,64,64))
print("Prediction : ",dataset.classes[np.argmax(pred.detach().cpu().numpy())] )
print("Prediction : ",pred.detach().cpu().numpy() )
