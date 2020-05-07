import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets

import torch.optim as optim

import tqdm

train = datasets.MNIST("", train= True, download= True,
                       transform= transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train= True, download= True,
                       transform= transforms.Compose([transforms.ToTensor()]))
                       
trainset = torch.utils.data.DataLoader(train, batch_size= 10, shuffle= True)
testset = torch.utils.data.DataLoader(test, batch_size= 10, shuffle= True)

def convert_y(y):
  y = y.numpy()
  shape = y.shape
  zeros = np.zeros((shape[0], 10))

  for idx, value in enumerate(y):
    zeros[idx, value] = 1
  return zeros

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 16, 3,stride= 1, padding= 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            )
        
        self.fc1 = nn.Linear(16 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(-1, 16 * 7 * 7)
        x = F.softmax(self.fc1(x))
        x = F.softmax(self.fc2(x))
        output = F.sigmoid(self.fc3(x))
        return output

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 512)
        self.fc3 = nn.Linear(512, 784)

    def forward(self, x):
        x = F.softmax(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return F.sigmoid(self.fc3(x))
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
generator = Generator()
discriminator = Discriminator()

# to GPU
generator.to(device)
discriminator.to(device)

# optimizers
priori_criterion = nn.MSELoss()
criterion = nn.BCELoss()

g_optimizer = optim.Adam(generator.parameters(), lr=0.00001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)

# ====== PRE-TRAINING OF GENERATOR ====== #
for epoch in range(5):
  for data in trainset:
    # Generator Train 
    X,y = data

    # y converting
    y = convert_y(y)
    y = torch.Tensor(y)
    y = y.to(device)

    generator.zero_grad()

    # predict
    output = generator(y)
    loss = priori_criterion(output.view(10 ,1 ,28 ,-1), X.to(device))
    loss.backward()
    g_optimizer.step()
  
  print(f'loss: {loss} ')


ALPHA = 10
for epoch in range(100):
  #start = datetime.now()

  for data in trainset:
    # Data generation
    X,y = data

    # X and y converting
    X = X.to(device)
    y = convert_y(y)
    y = torch.Tensor(y)
    y = y.to(device)

    # ========= DISCRIMINATOR TRAIN ============= #
    # real images
    discriminator.zero_grad()

    real_labels = torch.Tensor([0 for _ in range(10)]).to(device)
    output = discriminator(X)
    
    d_loss = criterion(output, real_labels)
    d_loss.backward()

    # fake images
    fake_image = generator(y)
    fake_labels = torch.Tensor([1 for _ in range(10)]).to(device)
    output = discriminator(fake_image.view(10,1,28,28))

    d_loss = criterion(output, fake_labels)
    d_loss.backward()
    d_optimizer.step()

    # ========= GENERATOR --> TRAIN ============= #
    generator.zero_grad()

    fake_image = generator(y)
    fake_labels = torch.Tensor([1 for _ in range(10)]).to(device)
    output = discriminator(fake_image.view(10,1,28,28))
    loss1 = criterion(output.view(-1), fake_labels)
    loss2 = criterion(fake_image.view(10,1,28,28),X)
    g_loss = loss2 + ALPHA * loss1
    g_loss.backward()
    g_optimizer.step()
  print(f'gloss: {g_loss}, dloss: {d_loss}')
