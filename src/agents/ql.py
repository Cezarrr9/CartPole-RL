import numpy as np
from collections import defaultdict
import gymnasium as gym

class QLAgent:

    def __init__(self,
                 action_space,
                 learning_rate: float, 
                 discount_factor: float, 
                 epsilon_start: float,
                 epsilon_end: float, 
                 epsilon_decay: float) -> None:
        
        self.action_space = action_space

        self.q_values = defaultdict(lambda: np.zeros(action_space.n))

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.training_error = []

    def select_action(self, obs: tuple[float, float, float, float]) -> int:

        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(self,
               obs: tuple[float, float, float, float],
               action: int,
               reward: int,
               terminated: bool,
               next_obs: tuple[float, float, float]) -> None:

        future_q_value = (not terminated) * np.max(self.q_values[tuple(next_obs)])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        self.q_values[obs][action] += self.learning_rate * temporal_difference

        self.training_error.append(temporal_difference)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)


        