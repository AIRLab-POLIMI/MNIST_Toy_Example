import torch
import torch.nn as nn

class classifier(nn.Module):
    def __init__(self,input_size,hidden_sizes,output_size = 10):
        super(classifier, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.projector = nn.Sequential(nn.Linear(self.input_size, self.hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
                        nn.ReLU())

        self.cls = nn.Sequential(nn.Linear(self.hidden_sizes[1], self.output_size),
                        nn.LogSoftmax(dim=1))
    
    def forward(self, x, return_embedding = False):
        embedding = self.projector(x)
        y  = self.cls(embedding)

        if return_embedding:
            return y, embedding
            
        return y