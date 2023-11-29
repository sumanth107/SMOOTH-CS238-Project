import numpy as np
import random
import collections
from TJunctionEnv import TJunctionEnv
from config import CONFIG


class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def choose_action(self):
        return self.env.action_space.sample()


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    n_steps = 10000
    RANDOM_RUNS = 1000
    results = collections.defaultdict(list)

    for _ in range(RANDOM_RUNS):
        env = TJunctionEnv(CONFIG)
        agent = RandomPolicy(env)

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
            action = agent.choose_action()
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
