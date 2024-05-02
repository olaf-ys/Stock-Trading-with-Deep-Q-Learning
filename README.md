# Stock-Trading-with-Deep-Q-Learning
-- Yuanshan Zhang, Mengxin Zhao, Jinke Han

## Introduction
Reinforcement learning is the state of art and most current AI research are focused on RL. It also succeeded in many real-world applications. For instance, auto-driving, the well-known Alpha Go, strategic gaming, and also stock trading. It is substantially different from supervised and unsupervised learning, and is regarded as the ‘third category of machine learning’ because in reinforcement learning, the agent learns by interacting with the environment.

DQN (Deep Q Network) is a combination of model-free reinforcement learning and deep neural network. DQN is suitable for tasks with large or continuous state or action space or when environment dynamics are indeterministic (reason for being a model-free algorithm). Therefore, using the action value function Q(s,a) enables the algorithm to learn the best policy directly from the action without the necessity of the full knowledge of the environment dynamics. 

DQN is therefore suitable in trading scenarios since the state space is large (the stock market is changing every second) and the full knowledge of the environment dynamics is impossible to gain (no one knows what tommorrows’ stock market will look like).

## What I did
The building of our DQN model consists of 4 major parts: 1. Environment  2. Model (MLP & LSTM) 3. Replay Memory 4. Agent

**1. Definations**\
I first define the following terminologies according to my purpose:
- Agent: the algorithm that can be trained to choose the optimal actions at each state to maximize the total rewards
- Environment: a stock trading simulation with which the agent can interact
- A (Action): hold, buy, and sell
- R (Reward): positive R when profit > 0, negative R when profit < 0
- S (State): stock info (given by a sliding window), balance, position, profits

**2. Building the environment**
First, an environment is built using the OpenAI gym to simulate stock trading. The environment consists of 3 crucial methods: 1. reset() 2. get_observation() 3. step() 4. render()

– the reset method restores the environment to the initial state at the beginning of a new episode. 
– the get_observation method concatenates the state information. 
– the step method assigns rewards and returns 3 components of the transition quintuple: next_state, reward, and done(if an episode is done). The render method renders real-time information to the screen. 

The functionality of the environment is tested using a subset of the data (see notebook for details)

