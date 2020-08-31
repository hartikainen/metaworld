import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


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
        right_finger_pos = self._get_site_pos('rightEndEffector')
        left_finger_pos = self._get_site_pos('leftEndEffector')
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
                'reach_success',
                'pick_success',
                'place_distance',
                'place_success')
        elif self.task_type == 'push':
            info_keys = (
                'reach_distance',
                'reach_success',
                'push_distance',
                'push_success')
        elif self.task_type == 'reach':
            info_keys = ('reach_distance', 'reach_success')
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
        ob = super().step(action)
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

    @property
    def _target_site_config(self):
        far_away = np.array([10., 10., 10.])
        return [
            ('goal_' + t, self._target_pos if t == self.task_type else far_away)
            for t in self.task_types
        ]

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('objGeom')

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos('objGeom')[-1]]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self._get_state_rand_vec()
        self.obj_init_pos = self.adjust_initObjPos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.object_height = self.data.get_geom_xpos('objGeom')[2]
        self.height_target = self.object_height + self.lift_threshold

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
            if self.task_type == 'push':
                self._target_pos = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
                self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            else:
                self._target_pos = goal_pos[-3:]
                self.obj_init_pos = goal_pos[:3]

        self._set_obj_xyz(self.obj_init_pos)
        self.max_reach_distance = np.linalg.norm(
            self.init_fingerCOM - np.array(self._target_pos),
            ord=2)
        self.max_push_distance = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2],
            ord=2)
        self.max_place_distance = np.linalg.norm((
            np.array([*self.obj_init_pos[:2], self.height_target])
            - np.array(self._target_pos)
        ), ord=2) + self.height_target

        self.num_resets += 1

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)
        self.init_fingerCOM = self.gripper_center_of_mass

    def compute_reward(self, actions, observation):
        object_position = observation['state_observation'][3:6]

        gripper_center_of_mass = self.gripper_center_of_mass

        height_target = self.height_target
        goal = self._target_pos

        def compute_reward_reach(actions, observation):
            del actions
            del observation

            reach_goal = self._target_pos

            reach_distance = np.linalg.norm(
                reach_goal - gripper_center_of_mass, ord=2)
            max_reach_distance = self.max_reach_distance

            max_reach_distance = self.max_reach_distance
            reach_distance = np.linalg.norm(gripper_center_of_mass - goal)

            reach_reward = (
                max_reach_distance - reach_distance
            ) / max_reach_distance
            success = reach_distance <= 5e-2

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
            assert np.all(goal == self._get_site_pos('goal_push'))

            object_position = observation['state_observation'][3:6]

            gripper_center_of_mass = self.gripper_center_of_mass
            push_goal = self._target_pos

            reach_distance = np.linalg.norm(
                object_position - gripper_center_of_mass, ord=2)
            max_reach_distance = self.max_reach_distance
            reach_success = reach_distance < 5e-2

            push_distance = np.linalg.norm(
                object_position[:2] - push_goal[:2], ord=2)
            max_push_distance = self.max_push_distance
            push_success = push_distance <= 7e-2

            max_reach_reward = max_reach_distance
            max_push_reward = max_push_distance

            reach_reward = (
                max_reach_reward
                if push_success
                else (max_reach_distance - reach_distance) / max_reach_distance)

            push_reward = (
                max_push_reward
                if push_success
                else (max_push_distance - push_distance) / max_push_distance)

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
            del observation

            reach_distance = reachDist = np.linalg.norm(object_position - gripper_center_of_mass)
            
            place_distance = placingDist = np.linalg.norm(object_position - goal)
            assert np.all(goal == self._get_site_pos('goal_pick_place'))

            def reachReward():
                epsilon = 1e-2
                max_reach_distance = self.max_reach_distance
                reach_distance_xy = np.linalg.norm(object_position[:-1] - gripper_center_of_mass[:-1])
                z_distance_from_reset = np.linalg.norm(
                    gripper_center_of_mass[-1] - self.init_fingerCOM[-1])
                reach_reward = (
                    - np.log(reach_distance + epsilon)
                    + np.log(max_reach_distance + epsilon))

                reward_bounds = [0.0, - np.log(epsilon)]

                if reach_distance < 5e-2:
                    reward = reach_reward + max(actions[-1], 0) / 25
                elif 5e-2 < reach_distance_xy:
                    reward = reach_reward + z_distance_from_reset

                return reach_reward, reach_distance

            def compute_pick_reward():
                object_geom_id = self.unwrapped.model.geom_name2id('objGeom')
                table_top_geom_id = self.unwrapped.model.geom_name2id('tableTop')
                leftpad_geom_id = self.unwrapped.model.geom_name2id('leftpad_geom')
                rightpad_geom_id = self.unwrapped.model.geom_name2id('rightpad_geom')

                table_top_object_contacts = [
                    x for x in self.unwrapped.data.contact
                    if (table_top_geom_id in (x.geom1, x.geom2)
                        and object_geom_id in (x.geom1, x.geom2))
                ]

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

                object_in_air = not table_top_object_contacts
                # pick_reward = float(gripping) * (1.0 + object_position[2])
                pick_reward_weight = (1 / self.lift_threshold)
                # pick_reward = (
                #     float(gripping)
                #     * float(object_in_air)
                #     * (1.0 - abs(self.height_target - object_position[2])))
                pick_success = gripping and object_in_air
                pick_reward = float(pick_success)
                # if (0 < leftpad_object_contact_force
                #     and 0 < rightpad_object_contact_force):
                #     total_gripper_object_contact_force = (
                #         leftpad_object_contact_force
                #         + rightpad_object_contact_force)
                #     target_contact_force = 800
    
                #     pick_reward_weight = 1e-2
                #     contact_force_reward = pick_reward_weight * min(
                #         total_gripper_object_contact_force, 500)
                # else:
                #     contact_force_reward = 0.0
                    
                # if not table_top_object_contacts:
                #     leftpad_object_contacts = [
                #         x for x in self.unwrapped.data.contact
                #         if (leftpad_geom_id in (x.geom1, x.geom2)
                #             and object_geom_id in (x.geom1, x.geom2))
                #     ]
        
                #     rightpad_object_contacts = [
                #         x for x in self.unwrapped.data.contact
                #         if (rightpad_geom_id in (x.geom1, x.geom2)
                #             and object_geom_id in (x.geom1, x.geom2))
                #     ]
        
                #     leftpad_object_contact_force = sum(
                #         self.unwrapped.data.efc_force[x.efc_address3]
                #         for x in leftpad_object_contacts)
        
                #     rightpad_object_contact_force = sum(
                #         self.unwrapped.data.efc_force[x.efc_address]
                #         for x in rightpad_object_contacts)
                    
                #     # print({
                #     #     'len(left_contacts)': len(leftpad_object_contacts),
                #     #     'len(right_contacts)': len(rightpad_object_contacts),
                #     #     'left-contacts': [round(self.unwrapped.data.efc_force[x.efc_address], 2)
                #     #              for x in leftpad_object_contacts],
                #     #     'right-contacts': [round(self.unwrapped.data.efc_force[x.efc_address], 2)
                #     #              for x in rightpad_object_contacts],
                #     #     'left': leftpad_object_contact_force,
                #     #     'right': rightpad_object_contact_force,
                #     # })
        
                #     if (0 < leftpad_object_contact_force
                #         and 0 < rightpad_object_contact_force):
                #         total_gripper_object_contact_force = (
                #             leftpad_object_contact_force
                #             + rightpad_object_contact_force)
                #         target_contact_force = 800
    
                #         pick_reward_weight = 1e-2
                #         contact_force_reward = pick_reward_weight * min(
                #             total_gripper_object_contact_force, 500)
                #     else:
                #         contact_force_reward = 0.0
                # else:
                #     contact_force_reward = 0.0

                return pick_reward, pick_success

            pick_reward, pick_success = compute_pick_reward()
            # pick_success = 0.0 < pick_reward 

            # def objDropped():
            #     return (object_position[2] < (self.object_height + 0.005)
            #             and 2e-2 < placingDist
            #             and 2e-2 < reachDist)
            #     # Object on the ground, far away from the goal, and from the gripper
            #     # Can tweak the margin limits

            def orig_pickReward():
                # height_target = self.height_target = self.object_height + self.lift_threshold
                hScale = 100
                if pick_success:
                    return hScale * height_target
                elif reachDist < 0.1 and (self.object_height + 0.005) < object_position[2]:
                    # objectDropped() or not self.pickCompleted
                    return hScale * min(height_target, object_position[2])
                else:
                    return 0

            def placeReward():
                cond = pick_success and reach_distance < 0.1
                if cond:
                    epsilon = 1e-2
                    max_place_distance = self.max_place_distance

                    place_reward_weight = (1 / max_place_distance) * 5.0
                    place_reward = place_reward_weight * (
                        max_place_distance - place_distance)
                    # place_reward = place_reward_weight * (
                    #     - np.log(place_distance + epsilon)
                    #     + np.log(max_place_distance + epsilon))
                else:
                    place_reward = 0.0

                # print({
                #     'pick_success': pick_success,
                #     'reachDist': round(reach_distance, 2),
                #     'cond': cond,
                #     'place_distance': round(place_distance, 2),
                #     'place_reward': round(place_reward, 2),
                # })

                return place_reward, place_distance

            # self.unwrapped.model.geom_name2id('leftpad_geom')
            #
            #
            # if 145 < self.curr_path_length:
            #     breakpoint()

            # grasp_contacts = [
            #     x for x in self.unwrapped.data.contact
            #     if 32 in (x.geom1, x.geom2) and 37 in (x.geom1, x.geom2)
            # ]
            reach_reward, reach_distance = reachReward()
            reach_success = reach_distance < 0.1
            # pick_reward = orig_pickReward()
            place_reward , place_distance = placeReward()
            # assert 0 <= place_reward and 0 <= pick_reward
            # assert 0 <= pick_reward, pick_reward
            reward = reach_reward + pick_reward + place_reward

            # print({
            #     'reward': round(reward, 2),
            #     'reach_rew': round(reach_reward, 2),
            #     'pick_rew': round(pick_reward, 2),
            #     'place_rew': round(place_reward, 2),
            #     'reach_dist': round(reach_distance, 2),
            #     'placing_dist': round(place_distance, 2),
            # })
            success = place_distance <= 0.07
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
