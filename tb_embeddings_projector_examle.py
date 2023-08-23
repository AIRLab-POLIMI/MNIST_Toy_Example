import os
from datetime import datetime
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch import nn
from torchvision import datasets, transforms
from itertools import islice
from torch import optim

# custom class for config file
from util.config import Config
from util.loggers import get_logger, get_tb_logger

# model class
from model import classifier


# loading experiment configuration
config = Config.load_config(os.path.join("configs","base_config.yml"))

# put here the name of the experiment you want to test
config.experiment_name = "baseline_230823_0844"
writer = get_tb_logger(config.tb_dir,config.experiment_name)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the data
valset = datasets.MNIST(config.dataset_folder, download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True)



# Build a feed-forward network
model = classifier(config.input_size,config.hidden_sizes,config.output_size)


# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(os.path.join(config.weigths_folder, config.experiment_name+".pth")))
model.to(device)
torch.set_grad_enabled(False)

# use only a subset of the validaiton set
limited_valloader = islice(valloader, config.n_embeddings_to_project)

# store the information we will pass to tensorboard in these lists
embeddings = list()
pred_labels = list()
true_labels = list()
corrects = list()
imgs = list()

# lets get some predictions
for images,labels in limited_valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        logps, embedding = model(img.cuda(),return_embedding = True)

        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])

        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]

        embeddings.append(embedding.cpu())
        pred_labels.append(pred_label)
        true_labels.append(true_label)
        imgs.append(images[i])

        if pred_label == true_label:
            corrects.append(True)
        else:
            corrects.append(False)


## record embeddings in tensorboard

# the mat tensor will contais all the embedding we want to project 
# must have shape (N,D), where N is number of data and D is feature dimension
embeddings_tensor = torch.cat(embeddings, dim=0)

# the image tensor must be of shape (N,C,H,W)
imgs_tensor = torch.cat(imgs, dim=0).unsqueeze(1)

# we want three filed in the metadata for each embedding
metadata_header = ["ground_truth", "prediction", "correct"]
# metadata is a list of N elements. Each elements will have `len(metadata_headr)` elements
metadata = [[true, pred, correct] for true, pred, correct in zip(true_labels, pred_labels, corrects)]


writer.add_embedding(mat = embeddings_tensor, 
                     metadata = metadata,
                     label_img = imgs_tensor, 
                     metadata_header = metadata_header)

