import numpy as np
import random
import collections
from TJunctionEnv import TJunctionEnv
from config import CONFIG
from utils import plot_test_performance



class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.0, epsilon_decay=0.99, min_epsilon=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize Q-table
        position_size = 201
        velocity_size = 16
        self.Q_table = np.zeros((position_size, velocity_size, env.action_space.n))

    def choose_action(self, state):
        av_position, av_velocity = int(state[0]), int(state[1])
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[av_position, av_velocity])

    def learn(self, state, action, reward, next_state, done):
        av_position, av_velocity = int(state[0]), int(state[1])
        next_av_position, next_av_velocity = int(next_state[0]), int(next_state[1])

        # Clamp the next state values to prevent out-of-bounds indexing
        next_av_position = min(max(next_av_position, 0), 200)
        next_av_velocity = min(max(next_av_velocity, 0), 15)

        old_value = self.Q_table[av_position, av_velocity, action]
        next_max = np.max(self.Q_table[next_av_position, next_av_velocity])

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                reward + self.discount_factor * next_max)
        self.Q_table[av_position, av_velocity, action] = new_value

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
                next_state = (int(next_state[0]), int(next_state[1]))  # Convert to discrete state

                self.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        plot_test_performance(rewards, 'qlearning_results.png')

        return rewards


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    n_steps = 1000
    RANDOM_RUNS = 5
    results = collections.defaultdict(list)

    for _ in range(RANDOM_RUNS):
        env = TJunctionEnv(CONFIG)
        agent = QLearning(env)
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
            if reward < -100:
                fatal = True
            if info['speeding']:
                speeding = True
            max_jerk = max(max_jerk, info['jerk'])
            if done:
                step_to_goal = step
                if (reward > 60) and (fatal is False):
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
