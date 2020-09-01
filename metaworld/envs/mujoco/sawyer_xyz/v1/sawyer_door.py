import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.utils import scaled_negative_log_reward


class SawyerDoorEnv(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0., 0.85, 0.1)
        obj_high = (0.1, 0.95, 0.1)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ]),
            'obj_init_pos': np.array([0.1, 0.95, 0.1]),
            'hand_init_pos': np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.2, 0.7, 0.15])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.door_angle_idx = self.model.get_joint_qpos_addr('doorjoint')

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_door_pull.xml')

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


    def _get_pos_objects(self):
        return self.data.get_geom_xpos('handle').copy()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_angle_idx] = pos
        qvel[self.door_angle_idx] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    def reset_model(self):
        self._reset_hand()

        self.objHeight = self.data.get_geom_xpos('handle')[2]

        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        self._target_pos = self.obj_init_pos + np.array([-0.3, -0.25, 0.05])

        self.sim.model.body_pos[self.model.body_name2id('door')] = self.obj_init_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos
        self._set_obj_xyz(0)
        self.max_reach_distance = np.linalg.norm(
            self.data.get_geom_xpos('handle') - self.init_fingerCOM,
            ord=2)
        self.max_pull_distance = np.linalg.norm(
            self.data.get_geom_xpos('handle')[:-1] - self._target_pos[:-1],
            ord=2)

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, actions, observation):
        del actions

        object_position = observation['state_observation'][3:6]

        gripper_center_of_mass = self.gripper_center_of_mass
        pull_goal = self._target_pos

        reach_distance = np.linalg.norm(
            object_position - gripper_center_of_mass, ord=2)
        max_reach_distance = self.max_reach_distance
        reach_success = reach_distance < 6.5e-2

        pull_distance = np.linalg.norm(
            object_position[:-1] - pull_goal[:-1], ord=2)
        max_pull_distance = self.max_pull_distance
        pull_success = pull_distance <= 8e-2

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
            reward_scale=2.0,
            epsilon=1e-2)

        reward = reach_reward + pull_reward
        success = pull_success

        result = {
            'reward': reward,
            'reach_reward': reach_reward,
            'reach_distance': reach_distance,
            'reach_success': reach_success,
            'pull_reward': pull_reward,
            'pull_distance': pull_distance,
            'pull_success': pull_success,
            'goal_distance': pull_distance,
            'success': success,
        }
        return result
