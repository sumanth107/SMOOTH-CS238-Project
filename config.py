CONFIG = {
    "actions_list": ["no_change", "speed_up", "speed_up_up", "slow_down", "slow_down_down"],
    "rewards_dict": {
        "goal_with_good_velocity": 40,
        "goal_with_bad_velocity": -40,
        "per_step_cost": -1,
        "over_speed_near_crosswalk": -20,
        "jerk_penalty_factor": 10,
    },
    "min_velocity": 0,
    "max_velocity": 20,
    "crosswalk_max_velocity": 2,
    "crosswalk_pos": 25,
    "init_velocity": 0,
    "max_position": 50,
    "init_state": [0, 3],
    "goal_state": [49, 10],
    "delta_time": 1}
