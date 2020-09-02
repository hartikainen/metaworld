import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


class SawyerButtonPressTopdownEnv(SawyerXYZEnv):

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.05)
        obj_high = (0.1, 0.9, 0.05)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.8, 0.05], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.88, 0.1])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.max_path_length = 150

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_button_press_topdown.xml')

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        ob = self._get_obs()
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
        position = self.data.get_site_xpos('buttonStart').copy()
        orientation = Rotation.from_matrix(
            self.data.get_site_xmat('buttonStart')).as_quat()
        velocity = self.data.get_site_xvelp('buttonStart').copy()
        return position, orientation, velocity

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = goal_pos
            button_pos = goal_pos.copy()
            button_pos[1] += 0.08
            button_pos[2] += 0.07
            self._state_goal = button_pos
            self._state_goal[2] -= 0.02

        self.sim.model.body_pos[self.model.body_name2id('box')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id('button')] = self._state_goal
        self._set_obj_xyz(0)
        self._state_goal = self.get_site_pos('hole')
        self.max_reach_distance = np.linalg.norm(
            self.data.get_site_xpos('buttonStart') - self.init_fingerCOM,
            ord=2)
        self.max_press_distance = np.abs(
            self.data.site_xpos[self.model.site_name2id('buttonStart')][2]
            - self._state_goal[2])

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
        press_goal = self._state_goal[2]

        reach_distance = np.linalg.norm(
            object_position - gripper_center_of_mass, ord=2)
        max_reach_distance = self.max_reach_distance
        reach_success = reach_distance < 5e-2

        press_distance = np.abs(object_position[2] - press_goal)
        max_press_distance = self.max_press_distance
        press_success = press_distance <= 2e-2

        reach_reward_weight = 1.0
        max_reach_reward = reach_reward_weight

        reach_reward = (
            max_reach_reward
            if press_success
            else (reach_reward_weight
                  * (max_reach_distance - reach_distance)
                  / max_reach_distance))

        press_reward_weight = 5.0
        press_reward = press_reward_weight * (
            max_press_distance - press_distance
        ) / max_press_distance

        reward = reach_reward + press_reward
        success = press_success

        result = {
            'reward': reward,
            'reach_reward': reach_reward,
            'reach_distance': reach_distance,
            'reach_success': reach_success,
            'press_reward': press_reward,
            'press_distance': press_distance,
            'press_success': press_success,
            'goal_distance': press_distance,
            'success': success,
        }
        return result
