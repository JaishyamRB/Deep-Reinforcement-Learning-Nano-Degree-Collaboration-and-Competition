# Udacity Deep Reinforcement Learning Nanodegree Project 3: Collaboration and Competition

## Project description

In this project, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).
Here are the Unity details of the environment:
```
Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: , 
```
## Getting Started

1. Before getting into the project there are certain dependencies to be met. Make sure you have [python 3.6]( https://www.python.org/downloads/release/python-3610/) installed and virtual environment.

2. Download the environment from one of the links below.
You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

3. Download or clone this repo and run these commands in your terminal:

```
pip install requirements.txt
pip -q install ./python
```

## Instructions

To run the agent open [Solution.ipynb](Solution.ipynb)

Description of the implementation is provided in [Report.md](Report.md). 
For technical details see the code.

The neural network architecture is implemented in [models.py](models.py)

The implementation of agents is in [Agent.py](Agent.py) which containes both DDPG and MADDPG agent. 

Common Replay buffer is implemented in [ReplayBuffer.pyReplayBuffer.py](ReplayBuffer.py)

Actor and critic model weights for Agent 1 is stored in [agent1_checkpoint_actor.pth](agent1_checkpoint_actor.pth) and [agent1_checkpoint_critic.pth](agent1_checkpoint_critic.pth),
and for Agent 2 at [agent2_checkpoint_actor.pth](agent2_checkpoint_actor.pth) and [agent2_checkpoint_critic.pth](agent2_checkpoint_critic.pth) respectively.
