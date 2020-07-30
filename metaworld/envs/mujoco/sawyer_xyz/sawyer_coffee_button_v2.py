import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


class SawyerCoffeeButtonEnvV2(SawyerXYZEnv):

    def __init__(self):

        self.max_push_dist = 0.04
        self.target_reward = 1000 * self.max_push_dist + 1000 * 2

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.)
        obj_high = (0.1, 0.9, 0.)
        goal_low = self._compute_goal(np.array(obj_low))[1]
        goal_high = self._compute_goal(np.array(obj_high))[1]

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.9, 0.]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .4, .2]),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 150

        self.obj_and_goal_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(goal_low, goal_high)
        self.observation_space = Box(
            np.hstack((self.hand_low, obj_low, obj_low, goal_low)),
            np.hstack((self.hand_high, obj_high, obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_coffee.xml', True)

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward, reach_dist, press_dist = self.compute_reward(ob)

        self.curr_path_length += 1
        info = {
            'reachDist': reach_dist,
            'goalDist': press_dist,
            'epRew': reward,
            'pickRew': None,
            'success': float(press_dist <= 0.02),
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_site_pos('buttonStart')

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('coffee_goal')] = goal[:3]

    def _compute_goal(self, init_obj_pos):
        """Given an initial object position, computes the goal position. The two
        positions may be mathematically related, but they don't have to be.

        Args:
            init_obj_pos (np.ndarray): Initial object position

        Returns:
            (bool, np.ndarray): A tuple that says (0) whether the initial obj
                pos and the goal are related mathematically and (1) the goal
                position
        """
        return True, init_obj_pos + np.array([.0, self.max_push_dist, .0])


    def reset_model(self):
        self._reset_hand()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']

        if self.random_init:
            self.obj_init_pos = self._get_state_rand_vec()

        self.sim.model.body_pos[self.model.body_name2id('coffee_machine')] = self.obj_init_pos
        self._set_obj_xyz(self.obj_init_pos)

        self._state_goal = self._compute_goal(self._get_pos_objects())[1]
        self._set_goal_marker(self._state_goal)
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(50):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1, 1], self.frame_skip)

    def compute_reward(self, obs):
        pos_button = obs[3:6]
        pos_finger_left = self.get_site_pos('leftEndEffector')

        press_dist = np.abs(pos_button[1] - self._state_goal[1])
        reach_dist = np.linalg.norm(pos_button - pos_finger_left)

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if reach_dist < 0.05:
            press_rew = 1000 * (self.max_push_dist - press_dist) + \
                        c1 * (np.exp(-(press_dist ** 2) / c2) +
                              np.exp(-(press_dist ** 2) / c3))
        else:
            press_rew = 0

        press_rew = max(press_rew, 0)
        reward = -reach_dist + press_rew

        return [reward, reach_dist, press_dist]
