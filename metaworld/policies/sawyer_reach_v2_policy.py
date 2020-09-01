import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerReachV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'hand_orientation': obs[3:7],
            'hand_velocity': obs[7:10],
            'puck_pos': obs[10:13],
            'puck_pos_padding': obs[13:16],
            'puck_orientation': obs[16:20],
            'puck_velocity': obs[20:23],
            'goal_pos': obs[23:26],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=o_d['goal_pos'], p=5.)
        action['grab_effort'] = 0.

        return action.array
