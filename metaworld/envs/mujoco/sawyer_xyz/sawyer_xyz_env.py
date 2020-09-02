import abc
import copy
import pickle

from gym.spaces import Box, Discrete
import mujoco_py
import numpy as np
from scipy.spatial.transform import Rotation

from metaworld.envs.mujoco.mujoco_env import MujocoEnv, _assert_task_is_set


class SawyerMocapBase(MujocoEnv, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    mocap for XYZ control.
    """
    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])

    def __init__(self, model_name, frame_skip=5):
        MujocoEnv.__init__(self, model_name, frame_skip=frame_skip)
        self.reset_mocap_welds()

    def get_endeffector_position_and_orientation(self):
        position = self.data.get_body_xpos('hand').copy()
        orientation = Rotation.from_matrix(
            self.data.get_body_xmat('hand').copy()).as_quat()
        velocity = self.data.get_body_xvelp('hand').copy()
        return position, orientation, velocity

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        del state['sim']
        del state['data']
        mjb = self.model.get_mjb()
        return {'state': state, 'mjb': mjb, 'env_state': self.get_env_state()}

    def __setstate__(self, state):
        self.__dict__ = state['state']
        self.model = mujoco_py.load_model_from_mjb(state['mjb'])
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.set_env_state(state['env_state'])

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()


class SawyerXYZEnv(SawyerMocapBase, metaclass=abc.ABCMeta):
    _HAND_SPACE = Box(
        np.array([-0.525, .35, -.0525] + [-1] * 4 + [-np.inf] * 3),
        np.array([+0.525, 1.025, .525] + [+1] * 4 + [+np.inf] * 3)
    )
    max_path_length = 200

    def __init__(
            self,
            model_name,
            frame_skip=5,
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.2, 0.75, 0.3),
            mocap_low=None,
            mocap_high=None,
            action_scale=1./100,
            action_rot_scale=1.,
    ):
        super().__init__(model_name, frame_skip=frame_skip)
        self.random_init = True
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)
        self.curr_path_length = 0
        self._freeze_rand_vec = True
        self._last_rand_vec = None

        # We use continuous goal space by default and
        # can discretize the goal space by calling
        # the `discretize_goal_space` method.
        self.discrete_goal_space = None
        self.discrete_goals = []
        self.active_discrete_goal = None

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
        )

        self._pos_obj_max_len = 6
        self._pos_obj_possible_lens = (3, 6)

        self._set_task_called = False
        self._partially_observable = True

        self.hand_init_pos = None  # OVERRIDE ME
        self._target_pos = None  # OVERRIDE ME
        self._random_reset_space = None  # OVERRIDE ME

    @property
    def gripper_center_of_mass(self):
        right_finger_pos = self._get_site_pos('rightEndEffector')
        left_finger_pos = self._get_site_pos('leftEndEffector')
        gripper_center_of_mass = (right_finger_pos + left_finger_pos) / 2.0
        return gripper_center_of_mass

    def _set_task_inner(self):
        # Doesn't absorb "extra" kwargs, to ensure nothing's missed.
        pass

    def set_task(self, task):
        self._set_task_called = True
        data = pickle.loads(task.data)
        assert isinstance(self, data['env_cls'])
        del data['env_cls']
        self._last_rand_vec = data['rand_vec']
        self._freeze_rand_vec = True
        self._last_rand_vec = data['rand_vec']
        del data['rand_vec']
        self._partially_observable = data['partially_observable']
        del data['partially_observable']
        self._set_task_inner(**data)

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]

        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def discretize_goal_space(self, goals):
        assert False
        assert len(goals) >= 1
        self.discrete_goals = goals
        # update the goal_space to a Discrete space
        self.discrete_goal_space = Discrete(len(self.discrete_goals))

    # Belows are methods for using the new wrappers.
    # `sample_goals` is implmented across the sawyer_xyz
    # as sampling from the task lists. This will be done
    # with the new `discrete_goals`. After all the algorithms
    # conform to this API (i.e. using the new wrapper), we can
    # just remove the underscore in all method signature.
    def sample_goals_(self, batch_size):
        assert False
        if self.discrete_goal_space is not None:
            return [self.discrete_goal_space.sample() for _ in range(batch_size)]
        else:
            return [self.goal_space.sample() for _ in range(batch_size)]

    def set_goal_(self, goal):
        assert False
        if self.discrete_goal_space is not None:
            self.active_discrete_goal = goal
            self.goal = self.discrete_goals[goal]
            self._target_pos_idx = np.zeros(len(self.discrete_goals))
            self._target_pos_idx[goal] = 1.
        else:
            self.goal = goal

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def _set_pos_site(self, name, pos):
        """Sets the position of the site corresponding to `name`

        Args:
            name (str): The site's name
            pos (np.ndarray): Flat, 3 element array indicating site's location
        """
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 1

        self.data.site_xpos[self.model.site_name2id(name)] = pos[:3]

    @property
    def _target_site_config(self):
        """Retrieves site name(s) and position(s) corresponding to env targets

        :rtype: list of (str, np.ndarray)
        """
        return [('goal', self._target_pos)]

    def _get_object_position_orientation_velocity(self):
        """Retrieves object position(s) from mujoco properties or instance vars

        Returns:
            np.ndarray: Flat array (usually 3 elements) representing the
                object(s)' position(s)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def _get_pos_goal(self):
        """Retrieves goal position from mujoco properties or instance vars

        Returns:
            np.ndarray: Flat array (3 elements) representing the goal position
        """
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos

    def _get_obs(self):
        """Combines positions of the end effector, object(s) and goal into a
        single flat observation

        Returns:
            np.ndarray: The flat observation array (12 elements)
        """
        (hand_position,
         hand_orientation,
         hand_velocity,
         ) = self.get_endeffector_position_and_orientation()

        (object_position,
         object_orientation,
         object_velocity,
         ) = self._get_object_position_orientation_velocity()
        assert len(object_position) in self._pos_obj_possible_lens
        # NOTE(hartikainen): All my changes brake with `len(pos_obj) != 3`
        assert len(object_position) == 3, object_position
        object_position_padded = np.zeros(self._pos_obj_max_len)
        object_position_padded[:len(object_position)] = object_position

        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)

        observation = np.hstack((
            hand_position,
            hand_orientation,
            hand_velocity,
            object_position_padded,
            object_orientation,
            object_velocity,
            pos_goal))

        return observation

    def _get_obs_dict(self):
        obs = self._get_obs()
        return dict(
            state_observation=obs,
            state_desired_goal=self._get_pos_goal(),
            state_achieved_goal=obs[3:-3],
        )

    @property
    def observation_space(self):
        object_position_low = np.full(6, -np.inf)
        object_position_high = np.full(6, +np.inf)
        object_orientation_low = np.full(4, -1.0)
        object_orientation_high = np.full(4, +1.0)
        object_velocity_low = np.full(3, -np.inf)
        object_velocity_high = np.full(3, +np.inf)
        goal_low = np.zeros(3) if self._partially_observable \
            else self.goal_space.low
        goal_high = np.zeros(3) if self._partially_observable \
            else self.goal_space.high
        return Box(
            np.hstack((
                self._HAND_SPACE.low,
                object_position_low,
                object_orientation_low,
                object_velocity_low,
                goal_low,
            )),
            np.hstack((
                self._HAND_SPACE.high,
                object_position_high,
                object_orientation_high,
                object_velocity_high,
                goal_high,
            ))
        )

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])

        for site in self._target_site_config:
            self._set_pos_site(*site)

        return self._get_obs()

    def reset(self):
        self.curr_path_length = 0
        return super().reset()

    def _reset_hand(self, steps=50):
        for _ in range(steps):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1, 1], self.frame_skip)

    def _get_state_rand_vec(self):
        if self._freeze_rand_vec:
            assert self._last_rand_vec is not None
            return self._last_rand_vec
        else:
            rand_vec = np.random.uniform(
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size)
            self._last_rand_vec = rand_vec
            return rand_vec
