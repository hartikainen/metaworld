from metaworld.policies.sawyer_pick_place_v2_policy import (
    SawyerPickPlaceV2Policy)

from metaworld.policies.sawyer_reach_v2_policy import (
    SawyerReachV2Policy)
from metaworld.policies.sawyer_push_v2_policy import (
    SawyerPushV2Policy)
from metaworld.policies.sawyer_pick_place_v2_policy import (
    SawyerPickPlaceV2Policy)
from metaworld.policies.sawyer_door_open_v2_policy import (
    SawyerDoorOpenV2Policy)
from metaworld.policies.sawyer_drawer_open_v2_policy import (
    SawyerDrawerOpenV2Policy)
from metaworld.policies.sawyer_drawer_close_v2_policy import (
    SawyerDrawerCloseV2Policy)
from metaworld.policies.sawyer_button_press_topdown_v2_policy import (
    SawyerButtonPressTopdownV2Policy)
from metaworld.policies.sawyer_peg_insertion_side_v2_policy import (
    SawyerPegInsertionSideV2Policy)
from metaworld.policies.sawyer_window_open_v2_policy import (
    SawyerWindowOpenV2Policy)
from metaworld.policies.sawyer_window_close_v2_policy import (
    SawyerWindowCloseV2Policy)

# from metaworld.policies.sawyer_reach_v1_policy import (
#     SawyerReachV1Policy)
# from metaworld.policies.sawyer_push_v1_policy import (
#     SawyerPushV1Policy)
# from metaworld.policies.sawyer_pick_place_v1_policy import (
#     SawyerPickPlaceV1Policy)
from metaworld.policies.sawyer_door_open_v1_policy import (
    SawyerDoorOpenV1Policy)
from metaworld.policies.sawyer_drawer_open_v1_policy import (
    SawyerDrawerOpenV1Policy)
from metaworld.policies.sawyer_drawer_close_v1_policy import (
    SawyerDrawerCloseV1Policy)
from metaworld.policies.sawyer_button_press_topdown_v1_policy import (
    SawyerButtonPressTopdownV1Policy)
# from metaworld.policies.sawyer_peg_insertion_side_v1_policy import (
#     SawyerPegInsertionSideV1Policy)
# from metaworld.policies.sawyer_window_open_v1_policy import (
#     SawyerWindowOpenV1Policy)
# from metaworld.policies.sawyer_window_close_v1_policy import (
#     SawyerWindowCloseV1Policy)

from metaworld.envs.mujoco.env_dict import (
    ALL_V2_ENVIRONMENTS, ALL_V1_ENVIRONMENTS)
from tests.metaworld.envs.mujoco.sawyer_xyz.utils import trajectory_summary


ENVIRONMENT_KEYS = (
    'reach-v2',
    'push-v2',
    'pick-place-v2',
    'door-open-v2',
    'drawer-open-v2',
    'drawer-close-v2',
    'button-press-topdown-v2',
    'peg-insert-side-v2',
    'window-open-v2',
    'window-close-v2',
)

POLICIES = {
    'reach-v2': SawyerReachV2Policy,
    'push-v2': SawyerPushV2Policy,
    'pick-place-v2': SawyerPickPlaceV2Policy,
    'door-open-v2': SawyerDoorOpenV2Policy,
    'drawer-open-v2': SawyerDrawerOpenV2Policy,
    'drawer-close-v2': SawyerDrawerCloseV2Policy,
    'button-press-topdown-v2': SawyerButtonPressTopdownV2Policy,
    'peg-insert-side-v2': SawyerPegInsertionSideV2Policy,
    'window-open-v2': SawyerWindowOpenV2Policy,
    'window-close-v2': SawyerWindowCloseV2Policy,

    'reach-v1': SawyerReachV2Policy,
    'push-v1': SawyerPushV2Policy,
    'pick-place-v1': SawyerPickPlaceV2Policy,
    'door-open-v1': SawyerDoorOpenV1Policy,
    'drawer-open-v1': SawyerDrawerOpenV1Policy,
    'drawer-close-v1': SawyerDrawerCloseV1Policy,
    'button-press-topdown-v1': SawyerButtonPressTopdownV1Policy,
    'peg-insert-side-v1': SawyerPegInsertionSideV2Policy,
    'window-open-v1': SawyerWindowOpenV2Policy,
    'window-close-v1': SawyerWindowCloseV2Policy,
}


def test_scripted_policy():
    # env_id = 'reach-v1'
    # env_id = 'push-v1'
    # env_id = 'pick-place-v1'
    # env_id = 'door-open-v1'
    # env_id = 'drawer-open-v1'
    # env_id = 'drawer-close-v1'
    # env_id = 'button-press-topdown-v1'
    env_id = 'peg-insert-side-v1'
    # env_id = 'window-open-v1'
    # env_id = 'window-close-v1'

    env = ALL_V1_ENVIRONMENTS[env_id]()
    if env_id == 'reach-v1':
        env._set_task_inner(task_type='reach')
    elif env_id == 'push-v1':
        env._set_task_inner(task_type='push')
    elif env_id == 'pick-place-v1':
        env._set_task_inner(task_type='pick_place')

    policy = POLICIES[env_id]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    action_space_ptp = env.action_space.high - env.action_space.low

    for i in range(100):
        env.reset()
        env.reset_model()
        o = env.reset()
        assert o.shape == env.observation_space.shape

        for _ in range(env.max_path_length):
            a = policy.get_action(o)
            # a = np.random.normal(a, act_noise_pct * action_space_ptp)

            o, r, done, info = env.step(a)

            env.render()

            if done:
                break

        print("\n\n")

test_scripted_policy()
