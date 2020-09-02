import numpy as np
from gym.spaces import  Box
from scipy.spatial.transform import Rotation

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerWindowOpenEnv(SawyerXYZEnv):

    def __init__(self):

        lift_threshold = 0.02
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.16)
        obj_high = (0.1, 0.9, 0.16)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([-0.1, 0.785, 0.15], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.08, 0.785, 0.15])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.lift_threshold = lift_threshold

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_window_horizontal.xml')

    @property
    def gripper_center_of_mass(self):
        right_finger_pos = self._get_site_pos('rightEndEffector')
        left_finger_pos = self._get_site_pos('leftEndEffector')
        gripper_center_of_mass = (right_finger_pos + left_finger_pos) / 2.0
        return gripper_center_of_mass

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        obs_dict = self._get_obs_dict()
        reward_info = self.compute_reward(action, obs_dict)
        reward = reward_info['reward']
        info = {
            **reward_info,
            'goal': self.goal,
        }
        terminal = False

        self.curr_path_length += 1

        return ob, reward, terminal, info

    def _get_object_position_orientation_velocity(self):
        position = self.data.get_site_xpos('handleOpenStart').copy()
        orientation = Rotation.from_matrix(
            self.data.get_site_xmat('handleOpenStart')).as_quat()
        velocity = self.data.get_site_xvelp('handleOpenStart').copy()
        return position, orientation, velocity

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.objHeight = self.data.get_geom_xpos('handle')[2]
        self.heightTarget = self.objHeight + self.lift_threshold

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy()
            goal_pos[0] += 0.18
            self._target_pos = goal_pos

        wall_pos = self.obj_init_pos.copy() - np.array([-0.1, 0, 0.12])
        window_another_pos = self.obj_init_pos.copy() + np.array([0.2, 0.03, 0])
        self.sim.model.body_pos[self.model.body_name2id('window')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id('window_another')] = window_another_pos
        self.sim.model.body_pos[self.model.body_name2id('wall')] = wall_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos
        self.max_pull_distance = 0.2
        self.max_reach_distance = np.linalg.norm(
            self.gripper_center_of_mass
            - self.data.get_geom_xpos('handle'),
            ord=2)

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        self.init_fingerCOM = self.gripper_center_of_mass

    def compute_reward(self, actions, observation):
        del actions

        object_position = observation['state_observation'][10:13]

        gripper_center_of_mass = self.gripper_center_of_mass
        pull_goal = self._target_pos

        reach_distance = np.linalg.norm(
            object_position - gripper_center_of_mass, ord=2)
        max_reach_distance = self.max_reach_distance
        reach_success = reach_distance < 5e-2

        pull_distance = np.linalg.norm(
            object_position[0] - pull_goal[0])
        max_pull_distance = self.max_pull_distance
        pull_success = pull_distance <= 5e-2

        reach_reward_weight = 1.0
        max_reach_reward = reach_reward_weight

        reach_reward = (
            max_reach_reward
            if pull_success
            else (reach_reward_weight
                  * (max_reach_distance - reach_distance)
                  / max_reach_distance))

        pull_reward_weight = 5.0
        pull_reward = pull_reward_weight * (
            max_pull_distance - pull_distance
        ) / max_pull_distance

        reward = reach_reward + pull_reward
        success = pull_success

        result = {
            'reward': reward,
            'reach_distance': reach_distance,
            'reach_reward': reach_reward,
            'reach_success': reach_success,
            'pull_distance': pull_distance,
            'pull_reward': pull_reward,
            'pull_success': pull_success,
            'success': success,
        }
        return result
