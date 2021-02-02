import torch
import torch.nn as nn
import pdb
from envs import *
from my_utils import *
from state_action import *
from torch.autograd import Variable

#----------------------------------------------
# Class of RL agent
#----------------------------------------------
class model(nn.Module):
    def __init__(self, a_hid_dim, c_hid_dim, env, device):
        super(model, self).__init__()        
        torch.set_default_dtype(torch.float64)
        def_act = def_action(env.env_parameter)
        def_sta = def_state(env.env_parameter)
        self.device = device
        self.N_UE = env.env_parameter.N_UE
        self.state_length = def_sta._ndarray_length()      
        self.num_actions = def_act.num_actions_valide
        self.num_type_of_action = def_act.num_type_of_action
        self.num_action_per_type = def_act.num_action_per_type
        self.num_actions_per_bw = self.num_action_per_type[0]*self.num_action_per_type[1]+self.num_action_per_type[0]
        
        # critic network
        self.critic = nn.Sequential(
            nn.Linear(self.state_length, c_hid_dim),
            nn.ReLU(),
            nn.Linear(c_hid_dim, c_hid_dim),
            nn.ReLU(),
            nn.Linear(c_hid_dim, c_hid_dim),
            nn.ReLU(),
            nn.Linear(c_hid_dim, 1)
        )
        
        # actor network
        self.actor = nn.Sequential(
            nn.Linear(self.state_length, a_hid_dim),
            nn.ReLU(),
            nn.Linear(a_hid_dim, a_hid_dim),
            nn.ReLU(),
            nn.Linear(a_hid_dim, a_hid_dim),
            nn.ReLU(),
            nn.Linear(a_hid_dim, self.num_actions),
            nn.ReLU(),
        )
        
    def forward(self, state_ndarray):
        # Input tensor, i.e. state
        state_tensor = torch.from_numpy(state_ndarray).double().to(self.device)
        state_tensor.requires_grad = True
        
        # Critic NN
        vpred = self.critic(state_tensor)
                                
        # Actor NN
        pi = self.actor(state_tensor)
        pi = nn.functional.softmax(pi,dim=0)
        pi = pi + 1e-15
        pi = pi/pi.sum()
        Is_UE_Not_D2D = 1-state_tensor[0:self.num_action_per_type[0]]
        Is_UE_Not_Tracking = state_tensor[self.num_action_per_type[0]:\
                                          (self.num_action_per_type[0] + self.num_action_per_type[1])]
        pi_mask = torch.zeros(self.num_actions, dtype=torch.float64, requires_grad=False).to(self.device)
        
        for i2 in range(self.num_action_per_type[2]): # loop of bw
            for i0 in range(self.num_action_per_type[0]): # loop of UE
                start_index = i2*self.num_actions_per_bw + i0*self.num_action_per_type[1]
                end_index = start_index + self.num_action_per_type[1] - 1
                pi_mask[start_index:end_index+1] = Is_UE_Not_D2D[i0]*Is_UE_Not_D2D*Is_UE_Not_Tracking[i0]*Is_UE_Not_Tracking
            pi_mask[end_index+1:end_index+1+self.num_action_per_type[0]] = Is_UE_Not_D2D*Is_UE_Not_Tracking
        pi = pi * pi_mask
        pi = pi + 1e-15
        pi = pi/pi.sum()
            
        # Sanity check of distribution
        if torch.isnan(pi.sum()) or abs(pi.sum() - 1) > 1e-5:
            pdb.set_trace()
            sys.exit('Wrong distribution in model_single_pi.py')
        
        # Return value function and policy
        return vpred, pi

    
#----------------------------------------------
# Save check point
#----------------------------------------------
def save_ckpt(epoch, path, loss, model, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, path)

    
#----------------------------------------------
# Load check point 
#----------------------------------------------
def load_ckpt(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    return epoch, loss, model, optimizers