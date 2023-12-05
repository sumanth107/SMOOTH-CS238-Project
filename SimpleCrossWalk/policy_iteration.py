import numpy as np
import random
import collections
from SimpleCrossWalkEnv import SmoothCrosswalkEnv
from config import CONFIG
from utils import plot_test_performance


class MonteCarloPolicyIteration:
    def __init__(self, env, discount_factor=0.95, epsilon=0.1):
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        position_size = int(env.observation_space.high[0]) + 1
        velocity_size = int(env.observation_space.high[1]) + 1
        self.Q = np.zeros((position_size, velocity_size, env.action_space.n))

        action_space = env.action_space.n

        self.returns = {(p, v): {a: [] for a in range(action_space)}
                        for p in range(position_size)
                        for v in range(velocity_size)}
        self.policy = np.zeros((position_size, velocity_size), dtype=int)

    def choose_action(self, state):
        # Epsilon-greedy policy
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            position, velocity = int(state[0]), int(state[1])
            return self.policy[position, velocity]

    def update_Q(self, episode):
        # Update action value function estimate using the episode
        states, actions, rewards = zip(*episode)
        discounts = np.array([self.discount_factor**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            position, velocity = int(state[0]), int(state[1])
            old_Q = self.Q[position, velocity, actions[i]]
            self.returns[(position, velocity)][actions[i]].append(sum(rewards[i:]*discounts[:-(i+1)]))
            self.Q[position, velocity, actions[i]] = np.mean(self.returns[(position, velocity)][actions[i]])

    def update_policy(self, state):
        # Update the policy based on the action-value function
        position, velocity = int(state[0]), int(state[1])
        self.policy[position, velocity] = np.argmax(self.Q[position, velocity])

    def generate_episode(self, env):
        # Generate an episode using the current policy
        episode = []
        state = env.reset()
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def train(self, env, num_episodes):
        rewards = []
        for itr in range(num_episodes):
            episode = self.generate_episode(env)
            total_reward = sum(e[2] for e in episode)
            self.update_Q(episode)
            for p in range(int(env.observation_space.high[0]) + 1):
                for v in range(int(env.observation_space.high[1]) + 1):
                    self.update_policy((p, v))
            rewards.append(total_reward)
            print(f"Episode {itr + 1}: Total Reward = {total_reward}")
        plot_test_performance(rewards, 'policyiteration_results.png')


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    n_steps = 10000
    RANDOM_RUNS = 5
    results = collections.defaultdict(list)

    for _ in range(RANDOM_RUNS):
        env = SmoothCrosswalkEnv(CONFIG)
        agent = MonteCarloPolicyIteration(env)
        agent.train(env, 4000)

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
