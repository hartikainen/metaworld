import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerReachPushPickPlaceEnv(SawyerXYZEnv):

    def __init__(self):
        liftThresh = 0.04
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

        self.liftThresh = liftThresh
        

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

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        (reward,
         reach_reward,
         reach_distance,
         _,
         push_distance,
         pick_reward,
         place_reward,
         place_distance) = self.compute_reward(action, ob)
        self.curr_path_length +=1

        goal_distance = (
            place_distance
            if self.task_type == 'pick_place'
            else push_distance)

        if self.task_type == 'reach':
            success = float(reach_distance <= 0.05)
        else:
            success = float(goal_distance <= 0.07)

        info = {
            'full_reward': reward,
            'reach_reward': reach_reward,
            'reach_distance': reach_distance,
            # _
            'push_distance': push_distance,
            'pick_reward': pick_reward,
            'place_reward': place_reward,
            'place_distance': place_distance,
            'goal_distance': goal_distance,
            'success': success,
        }
        info['goal'] = self.goal

        return ob, reward, False, info

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
        self.objHeight = self.data.get_geom_xpos('objGeom')[2]
        self.heightTarget = self.objHeight + self.liftThresh

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
        self.maxReachDist = np.linalg.norm(self.init_fingerCOM - np.array(self._target_pos))
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - np.array(self._target_pos)[:2])
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._target_pos)) + self.heightTarget
        self.target_rewards = [1000*self.maxPlacingDist + 1000*2, 1000*self.maxReachDist + 1000*2, 1000*self.maxPushDist + 1000*2]

        if self.task_type == 'reach':
            idx = 1
        elif self.task_type == 'push':
            idx = 2
        elif self.task_type == 'pick_place':
            idx = 0
        else:
            raise NotImplementedError

        self.target_reward = self.target_rewards[idx]
        self.num_resets += 1

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)
        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2

    def compute_reward(self, actions, obs):

        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        goal = self._target_pos

        def compute_reward_reach(actions, obs):
            del actions
            del obs

            epsilon = 1e-2
            max_reach_distance = self.maxReachDist
            reach_distance = np.linalg.norm(fingerCOM - goal)
            reach_reward = -np.log(
                reach_distance + epsilon) + np.log(max_reach_distance + epsilon)

            reward = reach_reward
            reachRew = reach_reward
            reachDist = reach_distance
            return [reward, reachRew, reachDist, None, None, None, None, None]

        def compute_reward_push(actions, obs):
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            del actions
            del obs

            assert np.all(goal == self._get_site_pos('goal_push'))
            reachDist = np.linalg.norm(fingerCOM - objPos)
            pushDist = np.linalg.norm(objPos[:2] - goal[:2])
            reachRew = -reachDist
            if reachDist < 0.05:
                pushRew = 1000*(self.maxPushDist - pushDist) + c1*(np.exp(-(pushDist**2)/c2) + np.exp(-(pushDist**2)/c3))
                pushRew = max(pushRew, 0)
            else:
                pushRew = 0
            reward = reachRew + pushRew
            return [reward, reachRew, reachDist, pushRew, pushDist, None, None, None]

        def compute_reward_pick_place(actions, obs):
            del obs

            reach_distance = reachDist = np.linalg.norm(objPos - fingerCOM)
            
            place_distance = placingDist = np.linalg.norm(objPos - goal)
            assert np.all(goal == self._get_site_pos('goal_pick_place'))

            def reachReward():
                epsilon = 1e-2
                max_reach_distance = self.maxReachDist
                reach_distance_xy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
                z_distance_from_reset = np.linalg.norm(
                    fingerCOM[-1] - self.init_fingerCOM[-1])
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
                # pick_reward = float(gripping) * (1.0 + objPos[2])
                pick_reward_weight = (1 / self.liftThresh) 
                # pick_reward = (
                #     float(gripping)
                #     * float(object_in_air)
                #     * (1.0 - abs(self.heightTarget - objPos[2])))
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
            #     return (objPos[2] < (self.objHeight + 0.005)
            #             and 2e-2 < placingDist
            #             and 2e-2 < reachDist)
            #     # Object on the ground, far away from the goal, and from the gripper
            #     # Can tweak the margin limits

            def orig_pickReward():
                # heightTarget = self.heightTarget = self.objHeight + self.liftThresh
                hScale = 100
                if pick_success:
                    return hScale * heightTarget
                elif reachDist < 0.1 and (self.objHeight + 0.005) < objPos[2]:
                    # objectDropped() or not self.pickCompleted
                    return hScale * min(heightTarget, objPos[2])
                else:
                    return 0

            def placeReward():
                cond = pick_success and reach_distance < 0.1
                if cond:
                    epsilon = 1e-2
                    max_place_distance = self.maxPlacingDist

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
            result = (
                reward,
                reach_reward,
                reach_distance,
                None,
                None,
                pick_reward,
                place_reward,
                place_distance)
            return result

        if self.task_type == 'reach':
            return compute_reward_reach(actions, obs)
        elif self.task_type == 'push':
            return compute_reward_push(actions, obs)
        elif self.task_type == 'pick_place':
            return compute_reward_pick_place(actions, obs)
        else:
            raise NotImplementedError
        
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:3] = [-0.2104139 ,  0.42137601, -0.31244928]
        self.viewer.cam.distance = 1.63
        self.viewer.cam.elevation = -35
        self.viewer.cam.azimuth = -140

        # print({
        #     'trackbodyid': self.viewer.cam.trackbodyid,
        #     'lookat': self.viewer.cam.lookat,
        #     'distance': self.viewer.cam.distance,
        #     'elevation': self.viewer.cam.elevation,
        #     'azimuth': self.viewer.cam.azimuth,
        #     'trackbodyid': self.viewer.cam.trackbodyid,
        # })

        # {
        #     'trackbodyid': -1,
        #     'lookat': array([-0.2104139 ,  0.42137601, -0.31244928]),
        #     'distance': 1.6286696771602396,
        #     'elevation': -35.87539776626939,
        #     'azimuth': -139.74854932301736
        # }
        # {
        #     'trackbodyid': -1,
        #     'lookat': array([-0.03778444,  0.26949207,  0.16464844]),
        #     'distance': 2.1741263994876223,
        #     'elevation': -25.903225806451488,
        #     'azimuth': -127.03225806451626
        # }
        return

