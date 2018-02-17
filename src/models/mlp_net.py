import torch
import torch.nn as nn
from torch.autograd import Variable

#Simple neural net with variable number of layers. 
#Uses SGD layers and relu activation

#inputs: (input_dimension, dutput_dimension, hidden_layers_width, number_of_hidden_layers)

class Net(nn.Module):
    def __init__(self, D_in, D_out,hidden_n, hidden_s):
        super(Net, self).__init__()  
        
        self.hidden_layers = hidden_n - 1
        self.hidden_s = hidden_s
        
        self.first_layer = torch.nn.Linear(D_in, hidden_s)
        self.middle_layers = nn.ModuleList()
        for i in range(self.hidden_layers):
            self.middle_layers.append(torch.nn.Linear(hidden_s, hidden_s))
        self.final_layer = torch.nn.Linear(hidden_s, D_out)
        
    def forward(self, x):
        h_relu = self.first_layer(x).clamp(min=0)
        for i in range(self.hidden_layers):
            h_relu = self.middle_layers[i](h_relu).clamp(min=0)   
        y_pred = self.final_layer(h_relu)
        return y_pred