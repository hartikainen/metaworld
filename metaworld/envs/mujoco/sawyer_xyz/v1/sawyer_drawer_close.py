import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerDrawerCloseEnv(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.04)
        obj_high = (0.1, 0.9, 0.04)
        goal_low = (-0.1, 0.699, 0.04)
        goal_high = (0.1, 0.701, 0.04)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0., 0.9, 0.04], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_drawer.xml')

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
        }
        terminal = False

        self.curr_path_length += 1

        return ob, reward, terminal, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('handle').copy()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.obj_init_pos - np.array([.0, .2, .0])
        self.objHeight = self.data.get_geom_xpos('handle')[2]

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy()
            goal_pos[1] -= 0.2
            self._target_pos = goal_pos

        drawer_cover_pos = self.obj_init_pos.copy()
        drawer_cover_pos[2] -= 0.02
        self.sim.model.body_pos[self.model.body_name2id('drawer')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id('drawer_cover')] = drawer_cover_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos
        self._set_obj_xyz(-0.2)
        self.max_pull_distance = np.abs(
            self.data.get_geom_xpos('handle')[1] - self._target_pos[1])
        self.max_reach_distance = np.linalg.norm(
            self.gripper_center_of_mass - self._target_pos,
            ord=2)

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)
        self.init_fingerCOM = self.gripper_center_of_mass

    def compute_reward(self, actions, observation):
        del actions

        object_position = observation['state_observation'][3:6]

        gripper_center_of_mass = self.gripper_center_of_mass
        pull_goal = self._target_pos

        reach_distance = np.linalg.norm(
            object_position - gripper_center_of_mass, ord=2)
        max_reach_distance = self.max_reach_distance
        reach_success = reach_distance < 5e-2

        pull_distance = np.linalg.norm(
            object_position[1] - pull_goal[1])
        max_pull_distance = self.max_pull_distance
        pull_success = pull_distance <= 6e-2

        epsilon = 1e-2
        max_reach_reward = -np.log(epsilon)
        max_pull_reward = -np.log(epsilon)

        reach_reward = (
            max_reach_reward
            if pull_success
            else (max_reach_distance - reach_distance) / max_reach_distance)

        pull_reward = (
            max_pull_reward
            if pull_success
            else (max_pull_distance - pull_distance) / max_pull_distance)

        reward = reach_reward + pull_reward
        success = pull_success

        return {
            'reward': reward,
            'reach_distance': reach_distance,
            'reach_reward': reach_reward,
            'reach_success': reach_success,
            'pull_distance': pull_distance,
            'pull_reward': pull_reward,
            'pull_success': pull_success,
            'success': success,
        }

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:3] = [-0.04424838, 0.59414413, 0.02608293]
        self.viewer.cam.distance = 0.93
        self.viewer.cam.elevation = -38.0
        self.viewer.cam.azimuth = -170.0

        return
