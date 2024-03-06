# CartPole-RL

## Overview
Hello everyone! This repository contains implementations of various reinforcement learning algorithms to solve the classic CartPole-v1 problem, a popular benchmark task in the field of reinforcement learning. The CartPole-v1 problem involves balancing a pole on a cart that moves along a frictionless track. The goal is to prevent the pole from falling over by moving the cart to the left or right.

## Algorithms Implemented
The following reinforcement learning algorithms have been implemented and applied to solve the CartPole-v1 problem:

- Q-Learning: An off-policy TD control algorithm that seeks to find the best action to take given the current state by learning the state action values.

- SARSA (State-Action-Reward-State-Action): An on-policy TD control algorithm that updates the state action values based on the action taken and the reward received, learning a policy that depends on the current action selection policy.

- REINFORCE: A policy gradient method that optimizes the policy directly. It updates policy parameters via gradient ascent on expected return.

- DQN (Deep Q-Network): An algorithm that combines Q-learning with deep neural networks to approximate state action values.

- DDQN (Double Deep Q-Network): An improvement over DQN that reduces the overestimation of the state action values by decoupling the target max operation into action selection and action evaluation.

## Results 
The performance of each algorithm is depicted in the *data* folder. The performance is measured in how long the CartPole managed to stay up without falling. After a maximum of 500 steps, the episode terminates by itself for CartPole-v1 from the gymnasium library. 
