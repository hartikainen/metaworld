import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerButtonPressTopdownV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'hand_orientation': obs[3:7],
            'hand_velocity': obs[7:10],
            'button_pos': obs[10:13],
            'button_pos_padding': obs[13:16],
            'button_orientation': obs[16:20],
            'button_velocity': obs[20:23],
            'goal_pos': obs[23:26],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = 1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_pos']

        button_z_goal = o_d['goal_pos'][-1]
        button_z_current = o_d['button_pos'][-1]

        if (np.linalg.norm(pos_curr[:2] - pos_button[:2]) > 0.04
            or np.abs(button_z_goal - button_z_current) < 2e-2):
            return pos_button + np.array([0., 0., 0.1])
        else:
            return pos_button
