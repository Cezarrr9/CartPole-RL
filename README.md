# CartPole-RL

## Overview
Hello everyone! This repository contains implementations of various reinforcement learning algorithms to solve the classic CartPole-v1 problem, a popular benchmark task in Reinforcement Learning. The CartPole-v1 problem involves balancing a pole on a cart that moves along a frictionless track. The goal is to prevent the pole from falling over by moving the cart to the left or right.

## Environment
All the information about the environment can be found here: https://gymnasium.farama.org/environments/classic_control/cart_pole/.

## Algorithms Implemented
The following reinforcement learning algorithms have been implemented and applied to solve the CartPole-v1 problem:

- Q-Learning

- SARSA
  
- REINFORCE

- DQN (Deep Q-Network)

- DDQN (Double Deep Q-Network)

## Results 
The performance of each algorithm is depicted in the *data* folder. The performance is measured in how long the CartPole managed to stay up without falling. After a maximum of 500 steps, the episode terminates by itself for CartPole-v1 from the gymnasium library. 

There are two types of figures. The figures named *name_of_the_algorithm.png* depict the algorithm's performance when seed = 1 is used. They show the raw performance, the average of the last 100 steps and the average of the previous steps at each point. The other figures show the algorithm's average performance over 5 seeds (1, 2, 3, 5 and 8) using a dark blue line. In these figures, the range of values generated at each step can also be observed, represented with a light shade of blue. 
