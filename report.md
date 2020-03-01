
# Continuous Control: Report

### 1. Introduction

This file provides a description of the implementation of 20 smart agents that can individually track a moving target. The report describes the learning algorithm, the choice of hyperparameters, the neural networks architecure and shows the performance of the jointly trained agents as the evolution of the rewards it obtaines over multiple episodic runs.

### 2. Implementation

#### 2.1 Learning Algorithm
The learning algorihtm is based on the Deep Deterministic Policy Gradient (DDPG) approach, where the hyperparameters have been adapted specifically to this problem.

The algorithm builds four networks, according to the DDPG approach: a local and a target network for the actor and a local and a target network for the critic. The actor network is based on the policy-gradient method so it maps a state into an action that during training should converge towards the optimum action for that state. The critic network is based on the Q-learning appoach and it computes the action-value function that should converge towards the optimum action-value function. In this implementation, the networks' architecture is described in the Network Architecture subsection below. 

To speed up training, 20 agents are used simultaneously. Each agent adds its experience to a replay buffer that is shared between all of 20 of them. The local actor and critic networks are updated after each time-step and for each of the 20 agents, concurrently using BATCH_SIZE different samples from the replay buffer of capacity BUFFER_SIZE. 

Since the target policy of the actor is deterministic, in order to make the agents try more options, noise is added to each action. This introduces an element of exploration in the action space (which is continuous, not discrete). The noise is generated as a Ornstein–Uhlenbeck process (which is time-correlated). 


*Local critic network update*
The target actor network predicts the optimal action, which is subsequently used by the target critic network to predict the state-action value function. Then an update on the local critic network is achieved through the computation of the mean-squared Bellman error (MSBE) between the two respective Q-functions (local and target) of the critic. The optimizer of choice is Adam with a learning rate of LR_CRITIC.

Update policy and value parameters using given batch of experience tuples: (state, action, reward, next_state, done)
Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) * (1 - done)
where:
  actor_target(state) -> action
  critic_target(state, action) -> Q-value
  
The Mean-squared Bellman error is defined by the Expectation[ (Q_locals - Q_targets)^2 ]


*Local actor network update*
The actor network learns in the basis of the local critic network through gradient ascent with a learning rate of LR_ACTOR. The local actor network predicts a best action *a* to take for a given state *s*, which is then used as an input to compute the total expected reward estimated using the local critic Q-value function. The negative of the total reward constitute the actor "loss" (hence performing gradient acent on the total reward).

*Target networks update*
The update of the target networks is performed via a parameter TAU such that: target_params = TAU*train_params + (1-TAU)*target_params.


#### 2.2 Network Architecture
The neural network (NN) models were design with two hidden layers of 256 nodes each. Each network has an input size of 33, corresponding to the dimension of the state space. The activation functions between layers are all ReLU. The actor network has an output size of 4 corresponding to the dimension of the action space. The output layer of the actor network is activated by *tanh* as it outputs values between -1 and 1 corresponding to the action space boundaries. The output layer of the critic network is unidimentional and linear (no activation function requried) as it outputs the total predicted reward for a given state and action.

#### 2.3 Hyperparameters Optimization

The list of hyperparameters of the current commit is as follows:

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay


### 3. Plot of Rewards

The following plot shows the joint training evoluiton of the 20 agents, over multiple episodes. The agents learn by episode 148 how to obtain an average reward (over 100 episodes) of at least +30. The reward is henceforth maintained.

![image](https://github.com/mionescu/udacity-continuous-control/blob/master/rewards_plot_v1.png)

*Rewards evolution over multiple episodes*

### 4. Discussion
In trying to improve the performance of the model, the number of nodes in the two hidden layers of the network was varied.
On one hand, trying 128 nodes for the hidden layer led to an average reward over 100 episodes of at least +30 after 160 episodes. On the other hand, trying 512 nodes on both hidden layers slowed the convergence significantly, the score being only 1.71 at the end of episode 100. The intermediate number of 256 nodes for the NNs proved to be a better choice.

The learning rates of both the actor and the critic networks were chosen to be 1e-3, since lower values made convergence slower.


### 5. Ideas for Future Work

To improve the agents' performance, some ideas for future work are to use gradient clipping when training the critic network, to play with the weight decay factor of the critic network and to implement other policy optimization approaches from this paper: https://arxiv.org/pdf/1604.06778.pdf.

