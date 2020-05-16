from ReplayBuffer import ReplayBuffer as memory
import numpy as np
from models import Actor, Critic
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate 


shared_memory = memory(BUFFER_SIZE, BATCH_SIZE)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Deep Deterministic Policy Gradient agent
class DDPGAgent():
    
    def __init__(self, state_size, action_size, seed, actor_file=None, critic_file=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            actor_file: path of file containing trained weights of actor network
            critic_file: path of file containing trained weights of critic network
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        
        #actor network:
        self.actor_local = Actor(state_size,action_size,seed).to(device)
        self.actor_target = Actor(state_size,action_size,seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(),LR)
        
        #critic network
        self.critic_local = Critic(state_size,action_size,seed).to(device)
        self.critic_target = Critic(state_size,action_size,seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(),LR)
        
        #load trained weights if needed
        if actor_file:
            weights = torch.load(actor_file)
            self.actor_local.load_state_dict(weights)
            self.actor_target.load_state_dict(weights)
            
        if critic_file:
            weights = torch.load(critic_file)
            self.critic_local.load_state_dict(weights)
            self.critic_target.load_state_dict(weights)
        
       
        
            
    def act(self, state):
        """Returns actions for given state as per current Actor network.
        
        Params
        ======
            state (array_like): current state
            
        """
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            
        self.actor_local.train()
        
        return np.clip(action,-1,1)
    
    def step(self):
        if len(shared_memory) > BATCH_SIZE:
            experiences = shared_memory.sample()
            self.learn(GAMMA,experiences)   

                
    def learn(self,GAMMA,experiences):
        """Update value parameters using batch of experience tuples.
        Params
        ======
            gamma (float): discount factor
        """
        
        states_list, actions_list, rewards, next_states_list, dones = experiences
                    
        next_states_tensor = torch.cat(next_states_list, dim=1).to(device)
        states_tensor = torch.cat(states_list, dim=1).to(device)
        actions_tensor = torch.cat(actions_list, dim=1).to(device)
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = [self.actor_target(next_states) for next_states in next_states_list]        
        next_actions_tensor = torch.cat(next_actions, dim=1).to(device)        
        Q_targets_next = self.critic_target(next_states_tensor, next_actions_tensor)        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))        
        # Compute critic loss
        Q_expected = self.critic_local(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(Q_expected, Q_targets)        
        # Minimize the loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # take the current states and predict actions
        actions_pred = [self.actor_local(states) for states in states_list]        
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(device)
        # -1 * (maximize) Q value for the current prediction
        actor_loss = -self.critic_local(states_tensor, actions_pred_tensor).mean()        
        # Minimize the loss
        self.actor_optim.zero_grad()
        actor_loss.backward()        
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optim.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

        
        
    def soft_update(self, local_model, target_model, tau=TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class MADDPGAgent:
    
    def __init__(self,state_size, action_size, seed,num_agents=2, actor_files=None, critic_files=None):
        
        self.num_agents = num_agents
        self.seed= seed
        self.state_size= state_size
        self.action_size= action_size
        
        if actor_files and critic_files :
            self.agents = [DDPGAgent(self.state_size, self.action_size, self.seed, actor_file=actor_files[i], critic_file=critic_files[i]) for i in range(self.num_agents)]
        else:
            self.agents = [DDPGAgent(self.state_size, self.action_size, self.seed) for i in range(self.num_agents)]
        
        
        
        
        
        
    def act(self,states):
        
        actions = np.zeros([self.num_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index,:] = agent.act(states[index])
        return actions
        
    def step(self, states,actions,rewards,next_states,dones):
        shared_memory.add(states, actions, rewards, next_states, dones)        
        for agent in self.agents:             
            agent.step()
     
    def save_weights(self):
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(index+1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(index+1))
            
              
            