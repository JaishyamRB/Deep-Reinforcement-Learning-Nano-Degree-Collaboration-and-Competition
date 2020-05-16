import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional  as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim

class Actor(nn.Module):
    
    def __init__(self, state_size, action_size,seed, fc1_size=512, fc2_size=256):
        """Initialize parameters and build model.
        Params
        ======
            seed (int): Random seed
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_size (int): Number of nodes in first fully connected layer
            fc2_size (int): Number of nodes in second fully connected layer
        """        
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        #Network Architecture
        self.fc1 = nn.Linear(state_size, fc1_size)
        #self.bn1 = nn.BatchNorm1d(fc1_size)
        self.fc2 = nn.Linear(fc1_size,fc2_size)
        #self.bn2 = nn.BatchNorm1d(fc2_size)
        self.fc3 = nn.Linear(fc2_size,action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self,state):      
        x= F.relu(self.fc1(state))
        x= F.relu(self.fc2(x))
        
        return F.tanh(self.fc3(x)) # to clip the logits from -1 to 1
        
        
class Critic(nn.Module):
    
    def __init__(self, state_size, action_size,seed, fc1_size=512, fc2_size=256):
        """Initialize parameters and build model.
        Params
        ======
            seed (int): Random seed
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_size (int): Number of nodes in first fully connected layer
            fc2_size (int): Number of nodes in second fully connected layer
        """        
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        #Network Architecture
        self.fc1 = nn.Linear((state_size+action_size)*2, fc1_size)
        #self.bn1 = nn.BatchNorm1d(fc1_size)
        self.fc2 = nn.Linear(fc1_size  ,fc2_size)
        self.fc3 = nn.Linear(fc2_size,1)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        xs = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(xs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)