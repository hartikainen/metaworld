import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.utils import scaled_negative_log_reward


class SawyerDrawerOpenEnv(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.04)
        obj_high = (0.1, 0.9, 0.04)
        goal_low = (-0.1, 0.5499, 0.04)
        goal_high = (0.1, 0.5501, 0.04)

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

        self.max_path_length = 150

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
        right_finger_pos = self.get_site_pos('rightEndEffector')
        left_finger_pos = self.get_site_pos('leftEndEffector')
        gripper_center_of_mass = (right_finger_pos + left_finger_pos) / 2.0
        return gripper_center_of_mass

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward_info = self.compute_reward(action, obs_dict)
        reward = reward_info['reward']
        info = {
            **reward_info,
        }
        terminal = False

        self.curr_path_length += 1

        return ob, reward, terminal, info

    def _get_object_position_orientation_velocity(self):
        position = self.data.get_geom_xpos('handle').copy()
        orientation = Rotation.from_matrix(
            self.data.get_geom_xmat('handle')).as_quat()
        velocity = self.data.get_geom_xvelp('handle').copy()
        return position, orientation, velocity

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = (self.get_site_pos('handleStart').copy() + self.data.get_geom_xpos('drawer_wall2').copy()) / 2
        return obs_dict

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.obj_init_pos - np.array([.0, .35, .0])
        self.objHeight = self.data.get_geom_xpos('handle')[2]

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy()
            goal_pos[1] -= 0.35
            self._state_goal = goal_pos

        self._set_goal_marker(self._state_goal)
        drawer_cover_pos = self.obj_init_pos.copy()
        drawer_cover_pos[2] -= 0.02
        self.sim.model.body_pos[self.model.body_name2id('drawer')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id('drawer_cover')] = drawer_cover_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._state_goal
        self.max_pull_distance = np.abs(
            self.data.get_geom_xpos('handle')[1] - self._state_goal[1])
        self.max_reach_distance = np.linalg.norm(
            self.gripper_center_of_mass - self.data.get_geom_xpos('handle'),
            ord=2)

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

        self.init_fingerCOM = self.gripper_center_of_mass

    def compute_reward(self, actions, observation):
        del actions

        object_position = observation['state_observation'][10:13]

        gripper_center_of_mass = self.gripper_center_of_mass
        pull_goal = self._state_goal

        reach_distance = np.linalg.norm(
            object_position - gripper_center_of_mass, ord=2)
        max_reach_distance = self.max_reach_distance
        reach_success = reach_distance < 5e-2

        pull_distance = np.linalg.norm(
            object_position[1] - pull_goal[1])
        max_pull_distance = self.max_pull_distance
        pull_success = pull_distance <= 6e-2

        reach_reward_weight = 1.0
        max_reach_reward = reach_reward_weight

        reach_reward = (
            max_reach_reward
            if pull_success
            else (reach_reward_weight
                  * (max_reach_distance - reach_distance)
                  / max_reach_distance))

        pull_reward = scaled_negative_log_reward(
            pull_distance,
            max_pull_distance,
            reward_scale=5.0,
            epsilon=1e-3)

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

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:3] = [-0.04424838, 0.59414413, 0.02608293]
        self.viewer.cam.distance = 0.93
        self.viewer.cam.elevation = -38.0
        self.viewer.cam.azimuth = -170.0

        return
