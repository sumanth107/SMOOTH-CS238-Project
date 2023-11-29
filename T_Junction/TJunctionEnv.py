import gym
from gym import spaces
import numpy as np


class TJunctionEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, config):
        super(TJunctionEnv, self).__init__()
        self.action_space = spaces.Discrete(5)  # ['yield', 'speed_up', 'speed_up_up', 'slow_down', 'slow_down_down']
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # [AV position, AV velocity, Car1 position, Car2 position]
            high=np.array([config['road_length_to_junction'] + config['road_length_after_junction'],
                           config['av_max_speed'],
                           config['main_road_length'],
                           config['main_road_length']]),
            dtype=np.float32
        )
        self.config = config
        self.prev_velocity = 0
        self.phase = 1  # Start with phase 1

    def step(self, action):
        done = False
        reward = 0
        jerk = 0
        speeding = False
        current_vel = self.state[1].copy()

        # Update positions of cars on main road
        self.state[2] -= self.config['car1_speed']
        self.state[3] -= self.config['car2_speed']

        # Phase 1: Approaching the middle (passing the first lane)
        if self.phase == 1:
            self._update_av_position(action)
            reward -= 1
            if 0 < self.state[2] < 25:
                if self.state[1] > 0:
                    reward -= self.config['fatal_penalty']
            if self.state[0] >= self.config['road_length_to_junction']:
                self.phase = 2  # Switch to phase 2

        # Phase 2: Turning left and merging
        elif self.phase == 2:
            reward -= 1
            if (abs(self.state[3]) < 15) and (action != 0):
                reward -= self.config['fatal_penalty']
            if action == 0:  # 'yield'
                reward -= self.config['yield_penalty']
            else:
                self.state[1] = self.config['av_turn_speed']
                reward += self.config['turning_reward']
                self.phase = 3  # Switch to phase 3

        # Phase 3: Traveling to destination
        elif self.phase == 3:
            self._update_av_position(action)
            reward -= 1
            self.state[0] += self.state[1]
            if self.state[0] >= self.config['road_length_to_junction'] + self.config['road_length_after_junction']:
                reward += self.config['reached_destination_reward']
                if self.state[1] > 10:
                    speeding = True
                    reward -= self.config['speeding']
                done = True
        jerk = self._calculate_jerk(self.prev_velocity, current_vel, self.state[1])
        self.prev_velocity = self.state[1]
        reward -= abs(jerk) * self.config['jerk_penalty_factor']

        return self.state, reward, done, {"speeding": speeding, "jerk": jerk}

    def reset(self):
        self.state = np.array([0, self.config['av_initial_speed'], self.config['car1_initial_position'],
                               self.config['car2_initial_position']])
        self.phase = 1
        return self.state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f"State: {self.state}, Phase: {self.phase}")

    def close(self):
        pass

    def _update_av_position(self, action):
        if action == 1:  # 'speed_up'
            self.state[1] = min(self.state[1] + 1, self.config['av_max_speed'])
        elif action == 2:  # 'speed_up_up'
            self.state[1] = min(self.state[1] + 2, self.config['av_max_speed'])
        elif action == 3:  # 'slow_down'
            self.state[1] = max(self.state[1] - 1, 0)
        elif action == 4:  # 'slow_down_down'
            self.state[1] = max(self.state[1] - 2, 0)
        self.state[0] += self.state[1]
        return

    def _calculate_jerk(self, prev_velocity, current_velocity, next_velocity):
        acceleration_t = current_velocity - prev_velocity
        acceleration_t_plus_1 = next_velocity - current_velocity
        jerk = (acceleration_t_plus_1 - acceleration_t)
        return jerk

    def _get_velocity_change(self, action):
        if action == 1:  # 'speed_up'
            return 1
        elif action == 2:  # 'speed_up_up'
            return 2
        elif action == 3:  # 'slow_down'
            return -1
        elif action == 4:  # 'slow_down_down'
            return -2
        return 0  # 'yield' or no change
