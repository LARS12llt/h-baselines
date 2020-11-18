"""Script containing the HierReplayBuffer object."""
import numpy as np
import random
import math
from functools import reduce


class HierReplayBuffer(object):
    """Hierarchical variant of ReplayBuffer.

    Attributes
    ----------
    buffer_size : int
        Max number of transitions to store in the buffer. When the buffer
        overflows the old memories are dropped.
    batch_size : int
        number of elements that are to be returned as a batch
    meta_period : int
        meta-policy action period
    obs_dim : int
        the number of elements in the observation
    ac_dim : int
        the number of elements in the environment action
    co_dim : int
        the number of elements in the context. Set to None if no context is
        used by the environment.
    goal_dim : int
        the number of elements in the meta-action
    num_levels : int
        the number of levels in the hierarchy
    """

    def __init__(self,
                 buffer_size,
                 batch_size,
                 meta_period,
                 obs_dim,
                 ac_dim,
                 co_dim,
                 goal_dim,
                 num_levels):
        """Instantiate the hierarchical replay buffer.

        Parameters
        ----------
        buffer_size : int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        batch_size : int
            number of elements that are to be returned as a batch
        meta_period : int
            meta-policy action period
        obs_dim : int
            the number of elements in the observation
        ac_dim : int
            the number of elements in the environment action
        co_dim : int
            the number of elements in the context. Set to None if no context is
            used by the environment.
        goal_dim : int
            the number of elements in the meta-action
        num_levels : int
            the number of levels in the hierarchy
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.meta_period = meta_period
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.co_dim = co_dim
        self.goal_dim = goal_dim
        self.num_levels = num_levels

        # some useful attributes
        self._size = 0
        self._current_idx = 0
        self._next_idx = 0
        self._sample_t = [[] for _ in range(buffer_size)]

    def __len__(self):
        """Return the number of elements stored."""
        return self._size

    def can_sample(self):
        """Check if n_samples samples can be sampled from the buffer.

        Returns
        -------
        bool
            True if enough sample exist, False otherwise
        """
        return len(self) >= self.batch_size

    def save(self, save_path):
        """Save parameters for the replay buffer."""
        np.save(save_path + '.obs_t.npy', self._obs_t)
        np.save(save_path + '.context_t.npy', self._context_t)
        np.save(save_path + '.action_t.npy', self._action_t)
        np.save(save_path + '.reward_t.npy', self._reward_t)
        np.save(save_path + '.done_t.npy', self._done_t)
        np.save(save_path + '.config.npy', np.array([
            self.buffer_size,
            self.batch_size,
            self.meta_period,
            self.obs_dim,
            self.ac_dim,
            self.co_dim,
            self.goal_dim,
            self.num_levels,
        ]))

    def load(self, save_path):
        """Load parameters for the replay buffer."""
        self._obs_t = np.load(save_path + '.obs_t.npy')
        self._context_t = np.load(save_path + '.context_t.npy')
        self._action_t = np.load(save_path + '.action_t.npy')
        self._reward_t = np.load(save_path + '.reward_t.npy')
        self._done_t = np.load(save_path + '.done_t.npy')
        (self.buffer_size,
         self.batch_size,
         self.meta_period,
         self.obs_dim,
         self.ac_dim,
         self.co_dim,
         self.goal_dim,
         self.num_levels) = np.load(save_path + '.config.npy')

    def is_full(self):
        """Check whether the replay buffer is full or not.

        Returns
        -------
        bool
            True if it is full, False otherwise
        """
        return len(self) == self.buffer_size

    def add(self, sample):
        """Add a new transition to the buffer.

        Parameters
        ----------
        sample : TODO
            TODO
        """
        self._sample_t[self._next_idx] = sample

        # Increment the next index and size terms
        self._current_idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, with_additional, collect_levels=None):
        """Sample a batch of experiences.

        An example for how a sample is collected from the list of observations/
        actions for a three-level hierarchy.

        Observations:

        ------------------------------------------
        | X  :   :   :   :   :   :   :   :   |   |     Level 2
        ------------------------------------------

        -----------------------------------------
        |   :   :   | X :   :   |   :   :   |   |     Level 1
        -----------------------------------------

        -----------------------------------------
        |   |   |   |   | X |   |   |   |   |   |     Level 0
        -----------------------------------------


        Next observations:

        -----------------------------------------
        |   :   :   :   :   :   :   :   :   |   |     Level 2
        -----------------------------------------

        -----------------------------------------
        |   :   :   |   :   :   | X :   :   |   |     Level 1
        -----------------------------------------

        -----------------------------------------
        |   |   |   |   |   | X |   |   |   |   |     Level 0
        -----------------------------------------


        Action:

        -----------------------------------------
        |     X     |           |           |   |     Level 2
        -----------------------------------------

        -----------------------------------------
        |   |   |   | X |   |   |   |   |   |   |     Level 1
        -----------------------------------------

        -----------------------------------------
        |   |   |   |   | X |   |   |   |   |   |     Level 0
        -----------------------------------------


        Reward:

        -------------------------------------
        |                 X                 |         Level 2
        -------------------------------------

        -------------------------------------
        |           |     X     |           |         Level 1
        -------------------------------------

        -------------------------------------
        |   |   |   |   | X |   |   |   |   |         Level 0
        -------------------------------------


        Context:

        -----------------------------------------
        |           |     X     |           |   |     Level 2 (action)
        -----------------------------------------

                            ^
                            | context

        -----------------------------------------
        |   |   |   |   | X |   |   |   |   |   |     Level 1 (action)
        -----------------------------------------

                            ^
                            | context

        -----------------------------------------
        |   |   |   |   | X |   |   |   |   |   |     Level 0 (action)
        -----------------------------------------


        Next Context:

        -----------------------------------------
        |           |           |     X     |   |     Level 2 (action)
        -----------------------------------------

                            ^
                            | context

        -----------------------------------------
        |   |   |   |   |   | x |   |   |   |   |     Level 1 (action)
        -----------------------------------------

                            ^
                            | context

        -----------------------------------------
        |   |   |   |   | X |   |   |   |   |   |     Level 0 (action)
        -----------------------------------------

        Parameters
        ----------
        with_additional : bool
            specifies whether to remove additional data from the replay buffer
            sampling procedure. Since only a subset of algorithms use
            additional data, removing it can speedup the other algorithms.
        collect_levels : list of int
            the levels in the hierarchy to collect data for. This is to avoid
            passing variables that are unused by the operation calling it.

        Returns
        -------
        list of array_like
            (batch_size, obs_dim) matrix of observations for every level in the
            hierarchy
        list of array_like
            (batch_size, obs_dim) matrix of next step observations for every
            level in the hierarchy
        list of array_like
            (batch_size, ac_dim) matrix of actions for every level in the
            hierarchy
        list of array_like
            (batch_size,) vector of rewards for every level in the hierarchy
        list of array_like
            (batch_size,) vector of done masks for every level in the hierarchy
        dict
            additional information; used for features such as the off-policy
            corrections or centralized value functions
        """
        meta_period = self.meta_period
        num_levels = self.num_levels
        collect_levels = collect_levels or list(range(num_levels))
        obses = [[] for _ in range(num_levels)]
        contexts = [[] for _ in range(num_levels)]
        actions = [[] for _ in range(num_levels)]
        next_obses = [[] for _ in range(num_levels)]
        next_contexts = [[] for _ in range(num_levels)]
        rewards = [[] for _ in range(num_levels)]
        dones = [[] for _ in range(num_levels)]

        # Do not encode additional information information in samples if it is
        # not needed. Waste of compute resources.
        if with_additional:
            worker_ob_dim = self.obs_dim + self.goal_dim
            worker_ac_dim = self.ac_dim
            additional = {
                "worker_obses": np.zeros(
                    (self.batch_size, worker_ob_dim, self.meta_period + 1),
                    dtype=np.float32),
                "worker_actions": np.zeros(
                    (self.batch_size, worker_ac_dim, self.meta_period),
                    dtype=np.float32),
            }
        else:
            additional = {}

        idxes = np.random.randint(0, self._size, size=self.batch_size)

        for k, indx in enumerate(idxes):
            # Extract the elements of the sample.
            candidate_sample = self._sample_t[indx]

            total_steps = len([None for sample in candidate_sample[-1]
                               if len(sample["observation"]) > 0])
            step = random.randint(0, total_steps - 1)

            # Collect samples for each level.
            for i in reversed(range(num_levels)):
                if i in collect_levels:
                    obses[i].append(candidate_sample[i][step]["observation"])
                    contexts[i].append(candidate_sample[i][step]["context"])
                    actions[i].append(candidate_sample[i][step]["action"])
                    next_obses[i].append(
                        candidate_sample[i][step]["next_observation"])
                    next_contexts[i].append(
                        candidate_sample[i][step]["next_context"])
                    rewards[i].append(candidate_sample[i][step]["reward"])
                    dones[i].append(candidate_sample[i][step]["done"])

                if isinstance(meta_period, list):
                    step = step // meta_period[i - 1]
                else:
                    step = step // meta_period

        # Convert everything to an array.
        for i in collect_levels:
            obses[i] = self._get_obs(obses[i], contexts[i], 1)
            next_obses[i] = self._get_obs(next_obses[i], next_contexts[i], 1)
            actions[i] = np.asarray(actions[i])
            rewards[i] = np.asarray(rewards[i])
            dones[i] = np.asarray(dones[i])

        return obses, next_obses, actions, rewards, dones, additional

    @staticmethod
    def _get_obs(obs, context, axis=0):
        """Return the processed observation.

        If the contextual term is not None, this will look as follows:

                                    -----------------
                    processed_obs = | obs | context |
                                    -----------------

        Otherwise, this method simply returns the observation.

        Parameters
        ----------
        obs : array_like
            the original observation
        context : array_like or None
            the contextual term. Set to None if no context is provided by the
            environment.
        axis : int
            the axis to concatenate the observations and contextual terms by

        Returns
        -------
        array_like
            the processed observation
        """
        obs = np.asarray(obs)
        if context[0] is not None:
            context = np.asarray(context)
            context = context.flatten() if axis == 0 else context
            obs = np.concatenate((obs, context), axis=axis)
        return obs
