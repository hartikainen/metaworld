import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerWindowOpenV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'hand_orientation': obs[3:7],
            'hand_velocity': obs[7:10],
            'wndw_pos': obs[10:13],
            'wndw_pos_padding': obs[13:16],
            'wndw_orientation': obs[16:20],
            'wndw_velocity': obs[20:23],
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
        pos_wndw = o_d['wndw_pos'] + np.array([-0.03, -0.03, -0.08])

        if np.linalg.norm(pos_curr[:2] - pos_wndw[:2]) > 0.04:
            return pos_wndw + np.array([0., 0., 0.3])
        elif abs(pos_curr[2] - pos_wndw[2]) > 0.02:
            return pos_wndw
        else:
            return pos_wndw + np.array([0.1, 0., 0.])
