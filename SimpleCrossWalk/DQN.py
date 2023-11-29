import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
from SimpleCrossWalkEnv import SmoothCrosswalkEnv
from config import CONFIG
from utils import plot_test_performance


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self, input_dim, output_dim, lr):
        self.dqn = DQN(input_dim, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.epsilon_decay = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.output_dim = output_dim

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_dim)
        q_values = self.dqn(state)
        return np.argmax(q_values.detach().numpy())

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            done = (done, )

        # Compute Q(s, a) - the model computes Q(s),
        # then we select the columns of actions taken
        q_values = self.dqn(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute Q(s_{t+1}) for all next states.
        next_q_values = self.dqn(next_state)
        next_q_value = next_q_values.max(1)[0]

        # Compute the expected Q values
        expected_q_value = reward + 0.99 * next_q_value * (1 - torch.tensor(done, dtype=torch.float))

        # Compute Huber loss
        loss = self.criterion(q_value, expected_q_value.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    n_steps = 1000
    RANDOM_RUNS = 5
    results = collections.defaultdict(list)

    for _ in range(RANDOM_RUNS):
        env = SmoothCrosswalkEnv(CONFIG)
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, lr=0.01)

        total_rewards = []
        for episode in range(n_steps):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        # Assuming you have a function called plot_test_performance
        plot_test_performance(total_rewards, 'dqn_results.png')

        avg_reward = sum(total_rewards) / n_steps
        print("Average Total Reward: {}".format(avg_reward))
