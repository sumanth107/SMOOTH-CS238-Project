import numpy as np
import random
import collections
from TJunctionEnv import TJunctionEnv
from config import CONFIG
from utils import plot_test_performance


class Q_Lambda:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, epsilon_decay=0.99, min_epsilon=0.01, lamb=0.9):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.lamb = lamb
        # Initialize Q-table and eligibility traces
        position_size = 201
        velocity_size = 16
        self.Q = np.zeros((position_size, velocity_size, env.action_space.n))
        self.N = np.zeros((position_size, velocity_size, env.action_space.n))

    def choose_action(self, state):
        # Using Epsilon-greedy policy
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore
            return self.env.action_space.sample()
        else:
            # Exploit
            position, velocity = int(state[0]), int(state[1])
            position = min(max(int(state[0]), 0), 200)
            velocity = min(max(int(state[1]), 0), 15)
            return np.argmax(self.Q[position, velocity])

    def learn(self, state, action, reward, next_state, done):
        position, velocity = int(state[0]), int(state[1])
        next_position, next_velocity = int(next_state[0]), int(next_state[1])
        next_position = min(max(next_position, 0), 200)
        next_velocity = min(max(next_velocity, 0), 15)

        # Ensure action is an integer
        action = int(action)

        # Choose the best action for next state for Q-learning
        next_action = np.argmax(self.Q[next_position, next_velocity])

        # Update Q-value and eligibility traces
        delta = reward + self.gamma * self.Q[next_position, next_velocity, next_action] - self.Q[position, velocity, action]
        self.N[position, velocity, action] += 1

        for pos in range(self.Q.shape[0]):
            for vel in range(self.Q.shape[1]):
                for act in range(self.Q.shape[2]):
                    self.Q[pos, vel, act] += self.alpha * delta * self.N[pos, vel, act]
                    self.N[pos, vel, act] = self.gamma * self.lamb * self.N[pos, vel, act]

        # Update epsilon
        if not done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            state = (int(state[0]), int(state[1]))  # Convert to discrete state
            total_reward = 0
            done = False

            # Reset eligibility traces
            self.N = np.zeros_like(self.N)

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = (int(next_state[0]), int(next_state[1]))
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        plot_test_performance(rewards, 'q_lambda_results.png')


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    n_steps = 1000
    RANDOM_RUNS = 5
    results = collections.defaultdict(list)

    for _ in range(RANDOM_RUNS):
        env = TJunctionEnv(CONFIG)
        agent = Q_Lambda(env)
        agent.train(1000)

        state = env.reset()
        total_reward = 0
        step_to_goal = 0
        total_jerk = 0
        max_jerk = -1
        speeding = False
        success = False
        fatal = False
        for step in range(n_steps):
            discrete_state = (int(state[0]), int(state[1]))
            action = agent.choose_action(discrete_state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_jerk += info['jerk']
            if info['speeding']:
                speeding = True
            if reward < -100:
                fatal = True
            max_jerk = max(max_jerk, info['jerk'])
            if done:
                step_to_goal = step
                if (reward > 60) and (fatal is False):
                    success = True
                break
        if done:
            results['total_reward'].append(total_reward)
            results['steps'].append(step_to_goal)
            results['total_jerk'].append(total_jerk)
            results['max_jerk'].append(max_jerk)
            results['speeding'].append(1 if speeding else 0)
            results['success'].append(1 if success else 0)
    print(results)
    avg_reward = sum(results['total_reward']) / RANDOM_RUNS
    print("Average Total Reward: {}".format(avg_reward))
    avg_steps = sum(results['steps']) / RANDOM_RUNS
    print("Average Steps: {}".format(avg_steps))
    sucess_rate = sum(results['success']) * 100 / RANDOM_RUNS
    print("Success Rate: {}%".format(sucess_rate))
    speeding_rate = sum(results['speeding']) * 100 / RANDOM_RUNS
    print("Speeding Rate: {}%".format(speeding_rate))