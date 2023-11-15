import numpy as np
import random
import collections
from SimpleCrossWalkEnv import SmoothCrosswalkEnv
from config import CONFIG
from utils import plot_test_performance


class SARSA:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.0, epsilon_decay=0.99, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        # Initialize Q-table
        position_size = int(env.observation_space.high[0]) + 1
        velocity_size = int(env.observation_space.high[1]) + 1
        self.Q = np.zeros((position_size, velocity_size, env.action_space.n))

    def choose_action(self, state):
        # Using Epsilon-greedy policy
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore
            return self.env.action_space.sample()
        else:
            # Exploit
            position, velocity = int(state[0]), int(state[1])
            return np.argmax(self.Q[position, velocity])

    def learn(self, state, action, reward, next_state, next_action, done):
        position, velocity = int(state[0]), int(state[1])
        next_position, next_velocity = int(next_state[0]), int(next_state[1])

        # Ensure action is an integer
        action = int(action)

        # Update Q-value
        new_value = self.alpha * (reward + self.gamma * self.Q[next_position, next_velocity, next_action] - self.Q[position, velocity, action])
        self.Q[position, velocity, action] += new_value

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

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                next_state = (int(next_state[0]), int(next_state[1]))
                self.learn(state, action, reward, next_state, next_action, done)
                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        plot_test_performance(rewards, 'sarsa_results.png')


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    n_steps = 10000
    RANDOM_RUNS = 5
    results = collections.defaultdict(list)

    for _ in range(RANDOM_RUNS):
        env = SmoothCrosswalkEnv(CONFIG)
        agent = SARSA(env)
        agent.train(4000)

        state = env.reset()
        total_reward = 0
        step_to_goal = 0
        total_jerk = 0
        max_jerk = -1
        speeding = False
        success = False
        for step in range(n_steps):
            discrete_state = (int(state[0]), int(state[1]))
            action = agent.choose_action(discrete_state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_jerk += info['jerk']
            if info['speeding']:
                speeding = True
            max_jerk = max(max_jerk, info['jerk'])
            if done:
                step_to_goal = step
                if reward == 39:
                    success = True
                break
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
