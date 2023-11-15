import numpy as np
import gym
from gym import spaces


class SmoothCrosswalkEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, config):
        super(SmoothCrosswalkEnv, self).__init__()

        self.actions_list = config['actions_list']
        self.action_space = spaces.Discrete(len(self.actions_list))
        self.config = config
        self.goal_state = config['goal_state']
        self.state = np.array([0, config['init_velocity']])
        self.prev_state = np.copy(self.state)
        self.crosswalk_max_velocity = config['crosswalk_max_velocity']

        self.observation_space = spaces.Box(
            low=np.array([0, config['min_velocity']]),
            high=np.array([config['max_position'], config['max_velocity']]),
            dtype=np.float32
        )

        self.prev_velocity = self.state[1]

        # Rewards
        self.rewards_dict = config['rewards_dict']

        # Environment parameters
        self.delta_time = 1
        self.crosswalk_pos = config['crosswalk_pos']

        # For jerk calculation
        self.jerk_penalty_factor = config.get('jerk_penalty_factor', 1)

    def step(self, action_index):
        speeding = False
        action = self.actions_list[action_index]
        self.prev_state = np.copy(self.state)

        velocity_change = self._get_velocity_change(action)
        new_velocity = np.clip(self.state[1] + velocity_change, 0, self.observation_space.high[1])
        new_position = np.clip(self.state[0] + new_velocity * self.delta_time, 0, self.observation_space.high[0])

        # Calculate jerk
        jerk = self._calculate_jerk(self.prev_velocity, self.state[1], new_velocity)
        self.prev_velocity = self.state[1]

        self.state = np.array([new_position, new_velocity])

        reward, done = self._calculate_reward(jerk)
        if self.prev_state[0] <= self.crosswalk_pos <= self.state[0]:
            if self.state[1] > self.crosswalk_max_velocity:
                speeding = True

        return self.state, reward, done, {"speeding": speeding, "jerk": jerk}

    def reset(self):
        self.state = np.array([0, self.config['init_velocity']])  # Reset to initial state
        self.prev_velocity = self.state[1]
        return self.state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass

    def _calculate_jerk(self, prev_velocity, current_velocity, next_velocity):
        acceleration_t = current_velocity - prev_velocity
        acceleration_t_plus_1 = next_velocity - current_velocity
        jerk = (acceleration_t_plus_1 - acceleration_t) / self.delta_time
        return jerk

    def _calculate_reward(self, jerk):
        reward = self.rewards_dict['per_step_cost']

        if self.state[0] >= self.goal_state[0]:
            if self.state[1] == self.goal_state[1]:
                reward += self.rewards_dict['goal_with_good_velocity']
            else:
                reward += self.rewards_dict['goal_with_bad_velocity']
            done = True
        else:
            done = False

        if self.prev_state[0] <= self.crosswalk_pos <= self.state[0]:
            if self.state[1] > self.crosswalk_max_velocity:
                reward += self.rewards_dict['over_speed_near_crosswalk']

        reward -= abs(jerk) * self.jerk_penalty_factor

        return reward, done

    def _get_velocity_change(self, action):
        if action == 'speed_up':
            return 1
        elif action == 'speed_up_up':
            return 2
        elif action == 'slow_down':
            return -1
        elif action == 'slow_down_down':
            return -2
        else:
            return 0
