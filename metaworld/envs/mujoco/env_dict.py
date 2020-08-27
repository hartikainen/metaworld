from collections import OrderedDict

from metaworld.envs.mujoco.sawyer_xyz import (
    SawyerNutAssemblyEnv,
    SawyerNutAssemblyEnvV2,
    SawyerBasketballEnv,
    SawyerBasketballEnvV2,
    SawyerBinPickingEnv,
    SawyerBinPickingEnvV2,
    SawyerBoxCloseEnv,
    SawyerBoxCloseEnvV2,
    SawyerButtonPressEnv,
    SawyerButtonPressTopdownEnv,
    SawyerButtonPressTopdownEnvV2,
    SawyerButtonPressTopdownWallEnv,
    SawyerButtonPressTopdownWallEnvV2,
    SawyerButtonPressEnvV2,
    SawyerButtonPressWallEnv,
    SawyerButtonPressWallEnvV2,
    SawyerCoffeeButtonEnv,
    SawyerCoffeeButtonEnvV2,
    SawyerCoffeePullEnv,
    SawyerCoffeePullEnvV2,
    SawyerCoffeePushEnv,
    SawyerCoffeePushEnvV2,
    SawyerDialTurnEnv,
    SawyerDialTurnEnvV2,
    SawyerNutDisassembleEnv,
    SawyerDoorEnv,
    SawyerDoorCloseEnv,
    SawyerDoorCloseEnvV2,
    SawyerDoorLockEnv,
    SawyerDoorLockEnvV2,
    SawyerDoorUnlockEnv,
    SawyerDoorUnlockEnvV2,
    SawyerDoorEnvV2,
    SawyerDrawerCloseEnv,
    SawyerDrawerCloseEnvV2,
    SawyerDrawerOpenEnv,
    SawyerDrawerOpenEnvV2,
    SawyerFaucetCloseEnv,
    SawyerFaucetCloseEnvV2,
    SawyerFaucetOpenEnv,
    SawyerFaucetOpenEnvV2,
    SawyerHammerEnv,
    SawyerHammerEnvV2,
    SawyerHandInsertEnv,
    SawyerHandInsertEnvV2,
    SawyerHandlePressEnv,
    SawyerHandlePressSideEnv,
    SawyerHandlePressSideEnvV2,
    SawyerHandlePressEnvV2,
    SawyerHandlePullEnv,
    SawyerHandlePullSideEnv,
    SawyerHandlePullSideEnvV2,
    SawyerHandlePullEnvV2,
    SawyerLeverPullEnv,
    SawyerLeverPullEnvV2,
    SawyerPegInsertionSideEnv,
    SawyerPegInsertionSideEnvV2,
    SawyerPegUnplugSideEnv,
    SawyerPickOutOfHoleEnv,
    SawyerPickOutOfHoleEnvV2,
    SawyerPickPlaceEnvV2,
    SawyerPickPlaceWallEnvV2,
    SawyerPlateSlideEnv,
    SawyerPlateSlideBackEnv,
    SawyerPlateSlideBackSideEnv,
    SawyerPlateSlideBackSideEnvV2,
    SawyerPlateSlideBackEnvV2,
    SawyerPlateSlideSideEnv,
    SawyerPlateSlideSideEnvV2,
    SawyerPlateSlideEnvV2,
    SawyerPushBackEnv,
    SawyerPushBackEnvV2,
    SawyerPushEnvV2,
    SawyerPushWallEnvV2,
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceWallEnv,
    SawyerReachEnvV2,
    SawyerReachWallEnvV2,
    SawyerShelfPlaceEnv,
    SawyerShelfPlaceEnvV2,
    SawyerSoccerEnv,
    SawyerSoccerEnvV2,
    SawyerStickPullEnv,
    SawyerStickPullEnvV2,
    SawyerStickPushEnv,
    SawyerStickPushEnvV2,
    SawyerSweepEnv,
    SawyerSweepEnvV2,
    SawyerSweepIntoGoalEnv,
    SawyerSweepIntoGoalEnvV2,
    SawyerWindowCloseEnv,
    SawyerWindowCloseEnvV2,
    SawyerWindowOpenEnv,
    SawyerWindowOpenEnvV2,
)


ALL_V1_ENVIRONMENTS = OrderedDict((
    ('reach-v1', SawyerReachPushPickPlaceEnv),
    ('push-v1', SawyerReachPushPickPlaceEnv),
    ('pick-place-v1', SawyerReachPushPickPlaceEnv),
    ('door-open-v1', SawyerDoorEnv),
    ('drawer-open-v1', SawyerDrawerOpenEnv),
    ('drawer-close-v1', SawyerDrawerCloseEnv),
    ('button-press-topdown-v1', SawyerButtonPressTopdownEnv),
    ('peg-insert-side-v1', SawyerPegInsertionSideEnv),
    ('window-open-v1', SawyerWindowOpenEnv),
    ('window-close-v1', SawyerWindowCloseEnv),
    ('door-close-v1', SawyerDoorCloseEnv),
    ('reach-wall-v1', SawyerReachPushPickPlaceWallEnv),
    ('pick-place-wall-v1', SawyerReachPushPickPlaceWallEnv),
    ('push-wall-v1', SawyerReachPushPickPlaceWallEnv),
    ('button-press-v1', SawyerButtonPressEnv),
    ('button-press-topdown-wall-v1', SawyerButtonPressTopdownWallEnv),
    ('button-press-wall-v1', SawyerButtonPressWallEnv),
    ('peg-unplug-side-v1', SawyerPegUnplugSideEnv),
    ('disassemble-v1', SawyerNutDisassembleEnv),
    ('hammer-v1', SawyerHammerEnv),
    ('plate-slide-v1', SawyerPlateSlideEnv),
    ('plate-slide-side-v1', SawyerPlateSlideSideEnv),
    ('plate-slide-back-v1', SawyerPlateSlideBackEnv),
    ('plate-slide-back-side-v1', SawyerPlateSlideBackSideEnv),
    ('handle-press-v1', SawyerHandlePressEnv),
    ('handle-pull-v1', SawyerHandlePullEnv),
    ('handle-press-side-v1', SawyerHandlePressSideEnv),
    ('handle-pull-side-v1', SawyerHandlePullSideEnv),
    ('stick-push-v1', SawyerStickPushEnv),
    ('stick-pull-v1', SawyerStickPullEnv),
    ('basketball-v1', SawyerBasketballEnv),
    ('soccer-v1', SawyerSoccerEnv),
    ('faucet-open-v1', SawyerFaucetOpenEnv),
    ('faucet-close-v1', SawyerFaucetCloseEnv),
    ('coffee-push-v1', SawyerCoffeePushEnv),
    ('coffee-pull-v1', SawyerCoffeePullEnv),
    ('coffee-button-v1', SawyerCoffeeButtonEnv),
    ('sweep-v1', SawyerSweepEnv),
    ('sweep-into-v1', SawyerSweepIntoGoalEnv),
    ('pick-out-of-hole-v1', SawyerPickOutOfHoleEnv),
    ('assembly-v1', SawyerNutAssemblyEnv),
    ('shelf-place-v1', SawyerShelfPlaceEnv),
    ('push-back-v1', SawyerPushBackEnv),
    ('lever-pull-v1', SawyerLeverPullEnv),
    ('dial-turn-v1', SawyerDialTurnEnv),
    ('bin-picking-v1', SawyerBinPickingEnv),
    ('box-close-v1', SawyerBoxCloseEnv),
    ('hand-insert-v1', SawyerHandInsertEnv),
    ('door-lock-v1', SawyerDoorLockEnv),
    ('door-unlock-v1', SawyerDoorUnlockEnv),))

ALL_V2_ENVIRONMENTS = OrderedDict((
    ('assembly-v2', SawyerNutAssemblyEnvV2),
    ('basketball-v2', SawyerBasketballEnvV2),
    ('bin-picking-v2', SawyerBinPickingEnvV2),
    ('box-close-v2', SawyerBoxCloseEnvV2),
    ('button-press-topdown-v2', SawyerButtonPressTopdownEnvV2),
    ('button-press-topdown-wall-v2', SawyerButtonPressTopdownWallEnvV2),
    ('button-press-v2', SawyerButtonPressEnvV2),
    ('button-press-wall-v2', SawyerButtonPressWallEnvV2),
    ('coffee-button-v2', SawyerCoffeeButtonEnvV2),
    ('coffee-pull-v2', SawyerCoffeePullEnvV2),
    ('coffee-push-v2', SawyerCoffeePushEnvV2),
    ('dial-turn-v2', SawyerDialTurnEnvV2),
    ('door-close-v2', SawyerDoorCloseEnvV2),
    ('door-lock-v2', SawyerDoorLockEnvV2),
    ('door-open-v2', SawyerDoorEnvV2),
    ('door-unlock-v2', SawyerDoorUnlockEnvV2),
    ('hand-insert-v2', SawyerHandInsertEnvV2),
    ('drawer-close-v2', SawyerDrawerCloseEnvV2),
    ('drawer-open-v2', SawyerDrawerOpenEnvV2),
    ('faucet-open-v2', SawyerFaucetOpenEnvV2),
    ('faucet-close-v2', SawyerFaucetCloseEnvV2),
    ('hammer-v2', SawyerHammerEnvV2),
    ('handle-press-side-v2', SawyerHandlePressSideEnvV2),
    ('handle-press-v2', SawyerHandlePressEnvV2),
    ('handle-pull-side-v2', SawyerHandlePullSideEnvV2),
    ('handle-pull-v2', SawyerHandlePullEnvV2),
    ('lever-pull-v2', SawyerLeverPullEnvV2),
    ('peg-insert-side-v2', SawyerPegInsertionSideEnvV2),
    ('pick-place-wall-v2', SawyerPickPlaceWallEnvV2),
    ('pick-out-of-hole-v2', SawyerPickOutOfHoleEnvV2),
    ('reach-v2', SawyerReachEnvV2),
    ('push-back-v2', SawyerPushBackEnvV2),
    ('push-v2', SawyerPushEnvV2),
    ('pick-place-v2', SawyerPickPlaceEnvV2),
    ('plate-slide-v2', SawyerPlateSlideEnvV2),
    ('plate-slide-side-v2', SawyerPlateSlideSideEnvV2),
    ('plate-slide-back-v2', SawyerPlateSlideBackEnvV2),
    ('plate-slide-back-side-v2', SawyerPlateSlideBackSideEnvV2),
    ('peg-insert-side-v2', SawyerPegInsertionSideEnvV2),
    ('soccer-v2', SawyerSoccerEnvV2),
    ('stick-push-v2', SawyerStickPushEnvV2),
    ('stick-pull-v2', SawyerStickPullEnvV2),
    ('push-wall-v2', SawyerPushWallEnvV2),
    ('push-v2', SawyerPushEnvV2),
    ('reach-wall-v2', SawyerReachWallEnvV2),
    ('reach-v2', SawyerReachEnvV2),
    ('shelf-place-v2', SawyerShelfPlaceEnvV2),
    ('sweep-into-v2', SawyerSweepIntoGoalEnvV2),
    ('sweep-v2', SawyerSweepEnvV2),
    ('window-open-v2', SawyerWindowOpenEnvV2),
    ('window-close-v2', SawyerWindowCloseEnvV2),
))

_NUM_METAWORLD_ENVS = len(ALL_V1_ENVIRONMENTS)

EASY_MODE_CLS_DICT = OrderedDict((
    ('reach-v1', SawyerReachPushPickPlaceEnv),
    ('push-v1', SawyerReachPushPickPlaceEnv),
    ('pick-place-v1', SawyerReachPushPickPlaceEnv),
    ('door-open-v1', SawyerDoorEnv),
    ('drawer-open-v1', SawyerDrawerOpenEnv),
    ('drawer-close-v1', SawyerDrawerCloseEnv),
    ('button-press-topdown-v1', SawyerButtonPressTopdownEnv),
    ('peg-insert-side-v1', SawyerPegInsertionSideEnv),
    ('window-open-v1', SawyerWindowOpenEnv),
    ('window-close-v1', SawyerWindowCloseEnv)),)


EASY_MODE_ARGS_KWARGS = {
    key: dict(args=[],
              kwargs={
                  'task_id': list(ALL_V1_ENVIRONMENTS.keys()).index(key)
              })
    for key, _ in EASY_MODE_CLS_DICT.items()
}

EASY_MODE_ARGS_KWARGS['reach-v1']['kwargs']['task_type'] = 'reach'
EASY_MODE_ARGS_KWARGS['push-v1']['kwargs']['task_type'] = 'push'
EASY_MODE_ARGS_KWARGS['pick-place-v1']['kwargs']['task_type'] = 'pick_place'


MEDIUM_MODE_CLS_DICT = OrderedDict((
    ('train',
        OrderedDict((
            ('reach-v1', SawyerReachPushPickPlaceEnv),
            ('push-v1', SawyerReachPushPickPlaceEnv),
            ('pick-place-v1', SawyerReachPushPickPlaceEnv),
            ('door-open-v1', SawyerDoorEnv),
            ('drawer-close-v1', SawyerDrawerCloseEnv),
            ('button-press-topdown-v1', SawyerButtonPressTopdownEnv),
            ('peg-insert-side-v1', SawyerPegInsertionSideEnv),
            ('window-open-v1', SawyerWindowOpenEnv),
            ('sweep-v1', SawyerSweepEnv),
            ('basketball-v1', SawyerBasketballEnv)))
    ),
    ('test',
        OrderedDict((
            ('drawer-open-v1', SawyerDrawerOpenEnv),
            ('door-close-v1', SawyerDoorCloseEnv),
            ('shelf-place-v1', SawyerShelfPlaceEnv),
            ('sweep-into-v1', SawyerSweepIntoGoalEnv),
            ('lever-pull-v1', SawyerLeverPullEnv,)))
    )
))
medium_mode_train_args_kwargs = {
    key: dict(args=[], kwargs={
        'task_id' : list(ALL_V1_ENVIRONMENTS.keys()).index(key),
    })
    for key, _ in MEDIUM_MODE_CLS_DICT['train'].items()
}

medium_mode_test_args_kwargs = {
    key: dict(args=[], kwargs={'task_id' : list(ALL_V1_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MEDIUM_MODE_CLS_DICT['test'].items()
}

medium_mode_train_args_kwargs['reach-v1']['kwargs']['task_type'] = 'reach'
medium_mode_train_args_kwargs['push-v1']['kwargs']['task_type'] = 'push'
medium_mode_train_args_kwargs['pick-place-v1']['kwargs'][
    'task_type'] = 'pick_place'

MEDIUM_MODE_ARGS_KWARGS = dict(
    train=medium_mode_train_args_kwargs,
    test=medium_mode_test_args_kwargs,
)
'''
    ML45 environments and arguments
'''
HARD_MODE_CLS_DICT = OrderedDict((
    ('train',
        OrderedDict((
            ('reach-v1', SawyerReachPushPickPlaceEnv),
            ('push-v1', SawyerReachPushPickPlaceEnv),
            ('pick-place-v1', SawyerReachPushPickPlaceEnv),
            ('door-open-v1', SawyerDoorEnv),
            ('drawer-open-v1', SawyerDrawerOpenEnv),
            ('drawer-close-v1', SawyerDrawerCloseEnv),
            ('button-press-topdown-v1', SawyerButtonPressTopdownEnv),
            ('peg-insert-side-v1', SawyerPegInsertionSideEnv),
            ('window-open-v1', SawyerWindowOpenEnv),
            ('window-close-v1', SawyerWindowCloseEnv),
            ('door-close-v1', SawyerDoorCloseEnv),
            ('reach-wall-v1', SawyerReachPushPickPlaceWallEnv),
            ('pick-place-wall-v1', SawyerReachPushPickPlaceWallEnv),
            ('push-wall-v1', SawyerReachPushPickPlaceWallEnv),
            ('button-press-v1', SawyerButtonPressEnv),
            ('button-press-topdown-wall-v1', SawyerButtonPressTopdownWallEnv),
            ('button-press-wall-v1', SawyerButtonPressWallEnv),
            ('peg-unplug-side-v1', SawyerPegUnplugSideEnv),
            ('disassemble-v1', SawyerNutDisassembleEnv),
            ('hammer-v1', SawyerHammerEnv),
            ('plate-slide-v1', SawyerPlateSlideEnv),
            ('plate-slide-side-v1', SawyerPlateSlideSideEnv),
            ('plate-slide-back-v1', SawyerPlateSlideBackEnv),
            ('plate-slide-back-side-v1', SawyerPlateSlideBackSideEnv),
            ('handle-press-v1', SawyerHandlePressEnv),
            ('handle-pull-v1', SawyerHandlePullEnv),
            ('handle-press-side-v1', SawyerHandlePressSideEnv),
            ('handle-pull-side-v1', SawyerHandlePullSideEnv),
            ('stick-push-v1', SawyerStickPushEnv),
            ('stick-pull-v1', SawyerStickPullEnv),
            ('basketball-v1', SawyerBasketballEnv),
            ('soccer-v1', SawyerSoccerEnv),
            ('faucet-open-v1', SawyerFaucetOpenEnv),
            ('faucet-close-v1', SawyerFaucetCloseEnv),
            ('coffee-push-v1', SawyerCoffeePushEnv),
            ('coffee-pull-v1', SawyerCoffeePullEnv),
            ('coffee-button-v1', SawyerCoffeeButtonEnv),
            ('sweep-v1', SawyerSweepEnv),
            ('sweep-into-v1', SawyerSweepIntoGoalEnv),
            ('pick-out-of-hole-v1', SawyerPickOutOfHoleEnv),
            ('assembly-v1', SawyerNutAssemblyEnv),
            ('shelf-place-v1', SawyerShelfPlaceEnv),
            ('push-back-v1', SawyerPushBackEnv),
            ('lever-pull-v1', SawyerLeverPullEnv),
            ('dial-turn-v1', SawyerDialTurnEnv),
        ))
    ),
    ('test',
        OrderedDict((
            ('bin-picking-v1', SawyerBinPickingEnv),
            ('box-close-v1', SawyerBoxCloseEnv),
            ('hand-insert-v1', SawyerHandInsertEnv),
            ('door-lock-v1', SawyerDoorLockEnv),
            ('door-unlock-v1', SawyerDoorUnlockEnv),
        ))
    )
))


def _hard_mode_args_kwargs(env_cls_, key_):
    del env_cls_

    kwargs = dict(task_id=list(ALL_V1_ENVIRONMENTS.keys()).index(key_))
    if key_ == 'reach-v1' or key_ == 'reach-wall-v1':
        kwargs['task_type'] = 'reach'
    elif key_ == 'push-v1' or key_ == 'push-wall-v1':
        kwargs['task_type'] = 'push'
    elif key_ == 'pick-place-v1' or key_ == 'pick-place-wall-v1':
        kwargs['task_type'] = 'pick_place'
    return dict(args=[], kwargs=kwargs)


HARD_MODE_ARGS_KWARGS = dict(train={}, test={})
for key, env_cls in HARD_MODE_CLS_DICT['train'].items():
    HARD_MODE_ARGS_KWARGS['train'][key] = _hard_mode_args_kwargs(env_cls, key)
for key, env_cls in HARD_MODE_CLS_DICT['test'].items():
    HARD_MODE_ARGS_KWARGS['test'][key] = _hard_mode_args_kwargs(env_cls, key)
