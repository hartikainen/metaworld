import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.utils import scaled_negative_log_reward


class SawyerReachPushPickPlaceEnv(SawyerXYZEnv):

    def __init__(self):
        lift_threshold = 0.04
        goal_low=(-0.1, 0.8, 0.05)
        goal_high=(0.1, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)

        self.task_types = ['pick_place', 'reach', 'push']

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.task_type = None
        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.lift_threshold = lift_threshold
        self.max_path_length = 150

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0

    def _set_task_inner(self, *, task_type, **kwargs):
        super()._set_task_inner(**kwargs)
        self.task_type = task_type

        # we only do one task from [pick_place, reach, push]
        # per instance of SawyerReachPushPickPlaceEnv.
        # Please only set task_type from constructor.
        if self.task_type == 'pick_place':
            self.goal = np.array([0.1, 0.8, 0.2])
        elif self.task_type == 'reach':
            self.goal = np.array([-0.1, 0.8, 0.2])
        elif self.task_type == 'push':
            self.goal = np.array([0.1, 0.8, 0.02])
        else:
            raise NotImplementedError

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_reach_push_pick_and_place.xml')

    @property
    def gripper_center_of_mass(self):
        right_finger_pos = self.get_site_pos('rightEndEffector')
        left_finger_pos = self.get_site_pos('leftEndEffector')
        gripper_center_of_mass = (right_finger_pos + left_finger_pos) / 2.0
        return gripper_center_of_mass

    def _visualize_success(self, info):
        reach_color = (
            [0, 1, 0, 1] if info['reach_success'] else [1, 0, 0, 1])
        self.model.site_rgba[
            self.model.site_name2id('leftEndEffector')
        ] = reach_color
        self.model.site_rgba[
            self.model.site_name2id('rightEndEffector')
        ] = reach_color

        if self.task_type in ('pick_place', 'push'):
            self.model.geom_rgba[self.model.geom_name2id('objGeom')] = (
                [0, 1, 0, 1]
                if info['success']
                else [1, 0, 0, 1])
        elif self.task_type == 'reach':
            pass
        else:
            raise ValueError(self.task_type)

    def _add_overlay(self, info):
        if getattr(self, 'viewer', None) is None:
            return

        if self.task_type == 'pick_place':
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
        elif self.task_type == 'push':
            info_keys = (
                'reach_distance',
                'push_distance',
                'reach_success',
                'push_success',
                'reach_reward',
                'push_reward',
                'reward')
        elif self.task_type == 'reach':
            info_keys = ('reach_distance', 'reach_success', 'reach_reward')
        else:
            raise ValueError(self.task_type)

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
        # The marker seems to get reset every time you do a simulation
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
        position = self.data.get_geom_xpos('objGeom').copy()
        orientation = Rotation.from_matrix(
            self.data.get_geom_xmat('objGeom')).as_quat()
        velocity = self.data.get_geom_xvelp('objGeom').copy()
        return position, orientation, velocity

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal_{}'.format(self.task_type))] = (
            goal[:3]
        )
        for task_type in self.task_types:
            if task_type != self.task_type:
                self.data.site_xpos[self.model.site_name2id('goal_{}'.format(task_type))] = (
                    np.array([10.0, 10.0, 10.0])
                )

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos('objGeom')[-1]]

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self._get_state_rand_vec()
        self.obj_init_pos = self.adjust_initObjPos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.object_height = self.data.get_geom_xpos('objGeom')[2]
        self.pick_height_target = self.object_height + self.lift_threshold

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._state_goal = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._state_goal[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._state_goal = goal_pos[3:]
            if self.task_type == 'push':
                self._state_goal = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
                self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            else:
                self._state_goal = goal_pos[-3:]
                self.obj_init_pos = goal_pos[:3]

        self._set_goal_marker(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos)
        self.max_reach_distance = np.linalg.norm(
            self.init_fingerCOM - np.array(self._state_goal),
            ord=2)
        self.max_push_distance = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._state_goal)[:2],
            ord=2)
        self.max_place_distance = np.linalg.norm((
            np.array([*self.obj_init_pos[:2], self.object_height])
            - np.array(self._state_goal)
        ), ord=2)

        self.num_resets += 1

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

        self.init_fingerCOM = self.gripper_center_of_mass

    @property
    def touching_object(self):
        object_geom_id = self.unwrapped.model.geom_name2id('objGeom')
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
        object_geom_id = self.unwrapped.model.geom_name2id('objGeom')
        table_top_geom_id = self.unwrapped.model.geom_name2id('tableTop')
        table_top_object_contacts = [
            x for x in self.unwrapped.data.contact
            if (table_top_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2))
        ]

        object_in_air = not table_top_object_contacts

        return object_in_air

    def compute_reward(self, actions, observation):
        def compute_reward_reach(actions, observation):
            del actions
            del observation

            gripper_center_of_mass = self.gripper_center_of_mass

            reach_goal = self._state_goal

            reach_distance = np.linalg.norm(
                reach_goal - gripper_center_of_mass, ord=2)
            max_reach_distance = self.max_reach_distance

            reach_reward_weight = 1.0
            reach_reward = reach_reward_weight * (
                max_reach_distance - reach_distance
            ) / max_reach_distance
            success = reach_distance < 5e-2

            result = {
                'reward': reach_reward,
                'reach_reward': reach_reward,
                'reach_distance': reach_distance,
                'reach_success': success,
                'goal_distance': reach_distance,
                'success': success,
            }
            return result

        def compute_reward_push(actions, observation):
            del actions

            object_position = observation['state_observation'][10:13]

            gripper_center_of_mass = self.gripper_center_of_mass
            push_goal = self._state_goal
            assert np.all(push_goal == self.get_site_pos('goal_push'))

            reach_distance = np.linalg.norm(
                object_position - gripper_center_of_mass, ord=2)
            max_reach_distance = self.max_reach_distance
            reach_success = reach_distance < 5e-2

            push_distance = np.linalg.norm(
                object_position[:2] - push_goal[:2], ord=2)
            max_push_distance = self.max_push_distance
            push_success = push_distance <= 7e-2

            reach_reward_weight = 1.0
            max_reach_reward = reach_reward_weight

            reach_reward = (reach_reward_weight
                            * (max_reach_distance - reach_distance)
                            / max_reach_distance)

            push_reward_weight = 5.0
            push_reward = float(reach_success) * push_reward_weight * (
                max_push_distance - push_distance
            ) / max_push_distance

            reward = reach_reward + push_reward
            success = push_success

            result = {
                'reward': reward,
                'reach_reward': reach_reward,
                'reach_distance': reach_distance,
                'reach_success': reach_success,
                'push_reward': push_reward,
                'push_distance': push_distance,
                'push_success': push_success,
                'goal_distance': push_distance,
                'success': success,
            }
            return result

        def compute_reward_pick_place(actions, observation):
            object_position = observation['state_observation'][10:13]

            gripper_center_of_mass = self.gripper_center_of_mass
            place_goal = self._state_goal
            assert np.all(place_goal == self.get_site_pos('goal_pick_place'))
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
            place_reward = place_reward_weight * (
                max_place_distance - place_distance
            ) / max_place_distance

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

        if self.task_type == 'reach':
            return compute_reward_reach(actions, observation)
        elif self.task_type == 'push':
            return compute_reward_push(actions, observation)
        elif self.task_type == 'pick_place':
            return compute_reward_pick_place(actions, observation)
        else:
            raise NotImplementedError
        
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:3] = [-0.04424838, 0.59414413, 0.02608293]
        self.viewer.cam.distance = 0.93
        self.viewer.cam.elevation = -38.0
        self.viewer.cam.azimuth = -170.0

        return
