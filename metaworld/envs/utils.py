import numpy as np


def scaled_negative_log_reward(observed_distance,
                               max_distance,
                               reward_scale=1.0,
                               epsilon=1e-2):
    """Computes a scaled negative log reward from distance.

    1. Compute unscaled_reward: `-log(observed_distance + epsilon)`
    2. Shift the reward between `[0, ...]` using `max_distance`
    3. Scale the reward between `[0, max_reward_scale]` using
       `reward_scale`
    """
    reward = - np.log(observed_distance + epsilon)
    reward += np.log(max_distance + epsilon)
    reward /= - np.log(epsilon) + np.log(max_distance + epsilon)
    reward *= reward_scale

    return reward
