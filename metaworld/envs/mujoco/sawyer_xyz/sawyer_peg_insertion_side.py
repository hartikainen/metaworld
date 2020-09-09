import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.utils import scaled_negative_log_reward


class SawyerPegInsertionSideEnv(SawyerXYZEnv):

    def __init__(self):

        lift_threshold = 0.11
        hand_init_pos = (0, 0.6, 0.2)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (.0, 0.5, 0.02)
        obj_high = (.2, 0.7, 0.02)
        goal_low = (-0.35, 0.4, -0.001)
        goal_high = (-0.25, 0.7, 0.001)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }
        self.goal = np.array([-0.3, 0.6, 0.0])

        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.lift_threshold = lift_threshold
        self.max_path_length = 150

        self.hand_init_pos = np.array(hand_init_pos)

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([.03, .0, .13]),
            np.array(goal_high) + np.array([.03, .0, .13])
        )

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_peg_insertion_side.xml')

    def _visualize_success(self, info):
        reach_color = (
            [0, 1, 0, 1] if info['reach_success'] else [1, 0, 0, 1])
        self.model.site_rgba[
            self.model.site_name2id('leftEndEffector')
        ] = reach_color
        self.model.site_rgba[
            self.model.site_name2id('rightEndEffector')
        ] = reach_color

        self.model.geom_rgba[self.model.geom_name2id('peg')] = (
            [0, 1, 0, 1]
            if info['success']
            else [1, 0, 0, 1])

    def _add_overlay(self, info):
        if getattr(self, 'viewer', None) is None:
            return

        info_keys = (
            'reach_distance',
            'place_distance',
            'reach_success',
            'pick_success',
            'place_success',
            'reach_reward',
            'pick_reward',
            'place_reward',
            'reward')

        if getattr(self.viewer, '_overlay', None) is not None:
            self.viewer._overlay.clear()

        for i, info_key in enumerate(info_keys):
            value = (
                str(info[info_key])
                if isinstance(info[info_key], (bool, np.bool, np.bool_))
                else str(round(info[info_key], 3)))
            self.viewer.add_overlay(0, info_key, value)

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])

        # Keep goal marker in place
        self._set_goal_marker(self._state_goal)

        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward_info = self.compute_reward(action, obs_dict)
        reward = reward_info['reward']
        info = {
            **reward_info,
            'goal': self.goal,
        }
        terminal = False

        self.curr_path_length +=1

        self._visualize_success(info)
        self._add_overlay(info)

        return ob, reward, terminal, info

    def _get_object_position_orientation_velocity(self):
        position = self.data.get_geom_xpos('peg').copy()
        orientation = Rotation.from_matrix(
            self.data.get_geom_xmat('peg')).as_quat()
        velocity = self.data.get_geom_xvelp('peg').copy()
        return position, orientation, velocity

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_goal_marker(self, pos):
        self.data.site_xpos[self.model.site_name2id('goal')] = pos

    def reset_model(self):
        self._reset_hand()

        pos_peg = self.obj_init_pos
        pos_box = self.goal

        if self.random_init:
            pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
            while np.linalg.norm(pos_peg[:2] - pos_box[:2]) < 0.1:
                pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)

        self.obj_init_pos = pos_peg
        self._set_obj_xyz(self.obj_init_pos)
        self.sim.model.body_pos[self.model.body_name2id('box')] = pos_box
        self._state_goal = pos_box + np.array([0.1, 0.0, 0.08])
        self._set_goal_marker(self._state_goal)

        self.object_height = self.obj_init_pos[2]
        self.pick_height_target = self.object_height + self.lift_threshold
        self.max_reach_distance = np.linalg.norm(
            self.init_fingerCOM - self.obj_init_pos, ord=2)
        self.max_place_distance = np.linalg.norm(
            self.obj_init_pos - self._state_goal, ord=2)

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(50):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1, 1], self.frame_skip)

        self.init_fingerCOM = self.gripper_center_of_mass

    @property
    def touching_object(self):
        object_geom_id = self.unwrapped.model.geom_name2id('peg')
        leftpad_geom_id = self.unwrapped.model.geom_name2id('leftpad_geom')
        rightpad_geom_id = self.unwrapped.model.geom_name2id('rightpad_geom')

        leftpad_object_contacts = [
            x for x in self.unwrapped.data.contact
            if (leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2))
        ]

        rightpad_object_contacts = [
            x for x in self.unwrapped.data.contact
            if (rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2))
        ]

        leftpad_object_contact_force = sum(
            self.unwrapped.data.efc_force[x.efc_address]
            for x in leftpad_object_contacts)

        rightpad_object_contact_force = sum(
            self.unwrapped.data.efc_force[x.efc_address]
            for x in rightpad_object_contacts)

        gripping = (0 < leftpad_object_contact_force
                    and 0 < rightpad_object_contact_force)

        return gripping

    @property
    def object_in_air(self):
        object_geom_id = self.unwrapped.model.geom_name2id('peg')
        table_top_geom_id = self.unwrapped.model.geom_name2id('tableTop')
        table_top_object_contacts = [
            x for x in self.unwrapped.data.contact
            if (table_top_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2))
        ]

        object_in_air = not table_top_object_contacts

        return object_in_air

    def compute_reward(self, actions, observation):
        object_position = observation['state_observation'][10:13]

        gripper_center_of_mass = self.gripper_center_of_mass
        place_goal = self._state_goal
        pick_height_target = self.pick_height_target

        reach_distance = np.linalg.norm(
            object_position - gripper_center_of_mass, ord=2)
        max_reach_distance = self.max_reach_distance
        reach_success = reach_distance < 1e-1

        place_distance = np.linalg.norm(object_position - place_goal, ord=2)
        max_place_distance = self.max_place_distance
        place_success = place_distance <= 7e-2

        reach_reward_weight = 1.0
        max_reach_reward = reach_reward_weight

        reach_reward = (
            max_reach_reward
            if place_success
            else (reach_reward_weight
                  * (max_reach_distance - reach_distance)
                  / max_reach_distance))

        touching_object = self.touching_object
        object_in_air = self.object_in_air

        pick_success = touching_object and object_in_air
        pick_reward_weight = 1.0
        pick_reward = (
            float(reach_success) * max(actions[-1], 0.0) / 10
            + pick_reward_weight * float(pick_success))

        place_reward_weight = 5.0
        place_distance_value = place_distance
        place_reward = float(object_in_air) * place_reward_weight * (
            (max_place_distance - place_distance) / max_place_distance
        ) + float(place_success) * place_reward_weight

        reward = reach_reward + pick_reward + place_reward
        success = place_success

        goal_distance = place_distance

        result = {
            'reward': reward,
            'reach_reward': reach_reward,
            'reach_distance': reach_distance,
            'reach_success': reach_success,
            'pick_reward': pick_reward,
            'pick_success': pick_success,
            'place_reward': place_reward,
            'place_distance': place_distance,
            'place_success': success,
            'goal_distance': goal_distance,
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
