import os
from datetime import datetime
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch import nn
from torchvision import datasets, transforms

from torch import optim

# custom class for config file
from util.config import Config
from util.loggers import get_logger, get_tb_logger


#loading experiment configuration
config = Config.load_config(os.path.join("configs","base_config.yml"))

# Adding the current time to experiment name to avoid over writes and confusion
config.experiment_name += "_"+datetime.now().strftime("%y%m%d_%H:%M")


# getting console logger and tensor board writer
writer = get_tb_logger(config.tb_dir,config.experiment_name)
logger = get_logger(config.log_dir,config.experiment_name)

logger.info(config.experiment_name)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST(config.dataset_folder, download=True, train=True, transform=transform)
valset = datasets.MNIST(config.dataset_folder, download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True)


# Layer details for the neural network
input_size = config.input_size
hidden_sizes = config.hidden_sizes
output_size = config.output_size

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
logger.info(model)

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

criterion = nn.NLLLoss()

# training loop
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
time0 = time()
for epoch in range(config.epochs):
    running_loss = 0
    for images, labels in trainloader:

        images = images.view(images.shape[0], -1)
    
        optimizer.zero_grad()
        
        output = model(images.cuda())
        loss = criterion(output, labels.cuda())
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()

    # logging the loss value
    epoch_loss = running_loss / len(trainloader) 
    writer.add_scalar('loss', epoch_loss, epoch)

logger.info(f"Training Time (in minutes) = {(time()-time0)/60}")

os.makedirs(config.weigths_folder, exist_ok = True)
torch.save(model.state_dict(), os.path.join(config.weigths_folder, config.experiment_name+'.pth'))
# to save both model and weigths
# torch.save(model, PATH)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             