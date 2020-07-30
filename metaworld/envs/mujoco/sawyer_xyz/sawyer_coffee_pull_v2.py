import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


class SawyerCoffeePullEnvV2(SawyerXYZEnv):

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.8, -0.001)
        obj_high = (0.05, 0.9, +0.001)
        goal_low = (-0.1, 0.5, -0.001)
        goal_high = (0.1, 0.6, +0.001)

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
        self._state_goal = np.array([0., 0.6, 0])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.obj_and_goal_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )

        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
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
        reward, reach_dist, pull_dist = self.compute_reward(ob)

        self.curr_path_length += 1
        info = {
            'reachDist': reach_dist,
            'goalDist': pull_dist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pull_dist <= 0.07),
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('mugbody')

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('mug_goal')] = (
            goal[:3]
        )

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('mug')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        #The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.get_body_com('obj')[-1]]

    def reset_model(self):
        self._reset_hand()
        # self.obj_init_pos = self.adjust_initObjPos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']

        if self.random_init:
            machine, goal = np.split(self._get_state_rand_vec(), 2)
            while np.linalg.norm(machine[:2] - goal[:2]) < 0.15:
                machine, goal = np.split(self._get_state_rand_vec(), 2)

            self.obj_init_pos = machine
            self._state_goal = goal

        self.sim.model.body_pos[self.model.body_name2id('coffee_machine')] = self.obj_init_pos
        self._set_obj_xyz(self.obj_init_pos)
        # self.sim.model.body_pos[self.model.body_name2id('mugbody')] = self.obj_init_pos + np.array([.0, -.05, .0])

        self.maxPullDist = np.linalg.norm(self.obj_init_pos[:2] - np.array(self._state_goal)[:2])

        self._set_goal_marker(self._state_goal)
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(50):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1, 1], self.frame_skip)

        self.init_fingers_center = self._get_fingers_center()

    def compute_reward(self, obs):
        pos_mug = obs[3:6]

        finger_center = self._get_fingers_center()

        goal = self._state_goal
        assert np.all(goal == self.get_site_pos('mug_goal'))

        pull_dist = np.linalg.norm(pos_mug[:2] - goal[:2])
        reach_dist = np.linalg.norm(finger_center - pos_mug)
        reach_distxy = np.linalg.norm(
            np.concatenate((
                pos_mug[:-1],
                [self.init_fingers_center[-1]]
            )) - finger_center
        )

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if reach_distxy < 0.05:
            reach_rew = -reach_dist + 0.1
        else:
            reach_rew = -reach_distxy

        if reach_dist < 0.05:
            pull_rew = 1000 * (self.maxPullDist - pull_dist) + \
                       c1 * (np.exp(-(pull_dist ** 2) / c2) +
                             np.exp(-(pull_dist ** 2) / c3))
        else:
            pull_rew = 0

        pull_rew = max(pull_rew, 0)
        reward = reach_rew + pull_rew

        return [reward, reach_dist, pull_dist]
