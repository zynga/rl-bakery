#
# Based on work found at:
# https://github.com/tensorflow/agents/blob/master/tf_agents/replay_buffers/py_replay_buffers_test.py
#

"""Unit tests for PyPrioritizedReplayBuffer."""

from __future__ import division
from __future__ import unicode_literals

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from rl_bakery.contrib.py_prioritized_replay_buffer import PyPrioritizedReplayBuffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import nest_utils

assert tf.executing_eagerly() is True, "Error: eager mode was not activate successfully"


class PyPrioritizedReplayBufferTest(parameterized.TestCase, tf.test.TestCase):

    def _create_replay_buffer(self, capacity=32):
        self._stack_count = 2
        self._single_shape = (1,)
        shape = (1, self._stack_count)
        observation_spec = array_spec.ArraySpec(shape, np.int32, 'obs')
        time_step_spec = ts.time_step_spec(observation_spec)
        action_spec = policy_step.PolicyStep(array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action'))
        self._trajectory_spec = trajectory.from_transition(
            time_step_spec, action_spec, time_step_spec)

        self._capacity = capacity
        self._alpha = 0.6
        self._replay_buffer = PyPrioritizedReplayBuffer(data_spec=self._trajectory_spec, capacity=self._capacity,
                                                        alpha=self._alpha)

    def _fill_replay_buffer(self, n_transition=50):
        # Generate N observations.
        single_obs_list = []
        obs_count = 100
        for k in range(obs_count):
            single_obs_list.append(np.full(self._single_shape, k, dtype=np.int32))

        # Add stack of observations to the replay buffer.
        time_steps = []
        for k in range(len(single_obs_list) - self._stack_count + 1):
            stacked_observation = np.concatenate(single_obs_list[k:k + self._stack_count], axis=-1)
            time_steps.append(ts.transition(stacked_observation, reward=0.0))

        self._experience_count = n_transition
        dummy_action = policy_step.PolicyStep(np.int32(0))
        for k in range(self._experience_count):
            self._replay_buffer.add_batch(nest_utils.batch_nested_array(trajectory.from_transition(time_steps[k],
                                                                                                   dummy_action,
                                                                                                   time_steps[k + 1])))

    def _generate_replay_buffer(self):
        self._create_replay_buffer()
        self._fill_replay_buffer()

    def testEmptyBuffer(self):
        self._create_replay_buffer()
        ds = self._replay_buffer.as_dataset(prioritized_buffer_beta=0.4, sample_batch_size=1)
        if tf.executing_eagerly():
            itr = iter(ds)
            mini_batch, indices, weights = next(itr)
        else:
            get_next = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
            res, indices, weights = self.evaluate(get_next)

        # make sure that the priority are set to 0 since the buffer is empty
        expected_priority = np.zeros((self._capacity,), dtype=np.float32)
        buffer_priority = self._replay_buffer._prioritized_buffer_priorities
        self.assertAllEqual(expected_priority, buffer_priority)

    def testEmptyBufferBatchSize(self):
        self._create_replay_buffer()
        ds = self._replay_buffer.as_dataset(sample_batch_size=2)
        if tf.executing_eagerly():
            next(iter(ds))
        else:
            get_next = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
            self.evaluate(get_next)

        # make sure that the priority are set to 0 since the buffer is empty
        expected_priority = np.zeros((self._capacity,), dtype=np.float32)
        buffer_priority = self._replay_buffer._prioritized_buffer_priorities
        self.assertAllEqual(expected_priority, buffer_priority)

    def validate_data(self, mini_batch, indices, weights=None):
        for idx, observation in enumerate(mini_batch.observation):
            observation_0 = observation[0][0]
            obs_index = indices[idx]
            self.assertAllEqual(observation_0, obs_index)

            if weights is not None:
                obs_weight = weights[idx]
                self.assertAllEqual(obs_weight, 1.0)

    def testReplayBufferFullDataset(self):
        np.random.seed(12345)

        buffer_size = 10
        self._create_replay_buffer(buffer_size)

        num_experiences = 10
        self._fill_replay_buffer(num_experiences)

        # make sure that the priority are set to 0 since the buffer is empty
        expected_priority = np.zeros((self._capacity,), dtype=np.float32)
        for i in range(num_experiences):
            if i >= self._capacity:
                break
            expected_priority[i] = 1.0

        buffer_priority = self._replay_buffer._prioritized_buffer_priorities
        self.assertAllEqual(expected_priority, buffer_priority)

        sample_batch_size = 1
        ds = self._replay_buffer.as_dataset(prioritized_buffer_beta=0.4, sample_batch_size=sample_batch_size)
        itr = iter(ds)
        mini_batch, indices, weights = next(itr)
        self.validate_data(mini_batch, indices, weights)

        sample_frequency = [0 for _ in range(10)]
        for i in range(10000):
            mini_batch, indices, weights = next(itr)
            for idx in indices:
                sample_frequency[idx] += 1

            if i % 100 == 0:
                self.validate_data(mini_batch, indices, weights)

        for i in range(10):
            self.assertAlmostEqual(10000 / 10, sample_frequency[i], delta=100)

    def testReplayBufferFullDatasetPrefetch(self):
        np.random.seed(12345)

        buffer_size = 10
        self._create_replay_buffer(buffer_size)

        num_experiences = 10
        self._fill_replay_buffer(num_experiences)

        # make sure that the priority are set to 0 since the buffer is empty
        expected_priority = np.zeros((self._capacity,), dtype=np.float32)
        for i in range(num_experiences):
            if i >= self._capacity:
                break
            expected_priority[i] = 1.0

        buffer_priority = self._replay_buffer._prioritized_buffer_priorities
        self.assertAllEqual(expected_priority, buffer_priority)

        sample_batch_size = 1
        prefetch_size = 10
        ds = self._replay_buffer.as_dataset(prioritized_buffer_beta=0.4, sample_batch_size=sample_batch_size).\
            prefetch(prefetch_size)
        itr = iter(ds)
        mini_batch, indices, weights = next(itr)
        self.validate_data(mini_batch, indices, weights)

        sample_frequency = [0 for _ in range(10)]
        for i in range(10000):
            mini_batch, indices, weights = next(itr)
            for idx in indices:
                sample_frequency[idx] += 1

            if i % 100 == 0:
                self.validate_data(mini_batch, indices, weights)

        for i in range(10):
            self.assertAlmostEqual(10000 / 10, sample_frequency[i], delta=100)

    def testReplayBufferFull(self):
        np.random.seed(12345)

        buffer_size = 10
        self._create_replay_buffer(buffer_size)

        num_experiences = 10
        self._fill_replay_buffer(num_experiences)

        # make sure that the priority are set to 0 since the buffer is empty
        expected_priority = np.zeros((self._capacity,), dtype=np.float32)
        for i in range(num_experiences):
            if i >= self._capacity:
                break
            expected_priority[i] = 1.0

        buffer_priority = self._replay_buffer._prioritized_buffer_priorities
        self.assertAllEqual(expected_priority, buffer_priority)

        sample_batch_size = 1
        mini_batch, indices, weights = self._replay_buffer.get_next(prioritized_buffer_beta=0.4,
                                                                    sample_batch_size=sample_batch_size)

        self.validate_data(mini_batch, indices, weights)

        sample_frequency = [0 for _ in range(10)]
        for i in range(10000):
            mini_batch, indices, weights = self._replay_buffer.get_next(prioritized_buffer_beta=0.4,
                                                                        sample_batch_size=sample_batch_size)
            if i % 100 == 0:
                self.validate_data(mini_batch, indices, weights)

            for idx in indices:
                sample_frequency[idx] += 1

        for i in range(10):
            self.assertAlmostEqual(10000 / 10, sample_frequency[i], delta=100)

    def testReplayBufferNotFull(self):

        np.random.seed(12345)

        buffer_size = 20
        self._create_replay_buffer(buffer_size)

        num_experiences = 10
        self._fill_replay_buffer(num_experiences)

        # make sure that the priority are set to 0 since the buffer is empty
        expected_priority = np.zeros((self._capacity,), dtype=np.float32)
        for i in range(num_experiences):
            if i >= self._capacity:
                break
            expected_priority[i] = 1.0

        buffer_priority = self._replay_buffer._prioritized_buffer_priorities
        self.assertAllEqual(expected_priority, buffer_priority)

        sample_batch_size = 1
        mini_batch, indices, weights = self._replay_buffer.get_next(prioritized_buffer_beta=0.4,
                                                                    sample_batch_size=sample_batch_size)

        self.validate_data(mini_batch, indices, weights)

        sample_frequency = [0 for _ in range(10)]
        for i in range(10000):
            mini_batch, indices, weights = self._replay_buffer.get_next(prioritized_buffer_beta=0.4,
                                                                        sample_batch_size=sample_batch_size)
            if i % 100 == 0:
                self.validate_data(mini_batch, indices, weights)

            for idx in indices:
                sample_frequency[idx] += 1

        for i in range(10):
            self.assertAlmostEqual(10000 / 10, sample_frequency[i], delta=100)

    def testReplayBufferBatchSize(self):

        np.random.seed(12345)

        buffer_size = 20
        self._create_replay_buffer(buffer_size)

        num_experiences = 10
        self._fill_replay_buffer(num_experiences)

        # make sure that the priority are set to 0 since the buffer is empty
        expected_priority = np.zeros((self._capacity,), dtype=np.float32)
        for i in range(num_experiences):
            if i >= self._capacity:
                break
            expected_priority[i] = 1.0

        buffer_priority = self._replay_buffer._prioritized_buffer_priorities
        self.assertAllEqual(expected_priority, buffer_priority)

        sample_batch_size = 10
        mini_batch, indices, weights = self._replay_buffer.get_next(prioritized_buffer_beta=0.4,
                                                                    sample_batch_size=sample_batch_size)

        self.validate_data(mini_batch, indices, weights)

        sample_frequency = [0 for _ in range(10)]
        for i in range(1000):
            mini_batch, indices, weights = self._replay_buffer.get_next(prioritized_buffer_beta=0.4,
                                                                        sample_batch_size=sample_batch_size)
            if i % 100 == 0:
                self.validate_data(mini_batch, indices, weights)

            for idx in indices:
                sample_frequency[idx] += 1

        for i in range(10):
            self.assertAlmostEqual(10000 / 10, sample_frequency[i], delta=100)

    def testPrioritizedReplayBuffer(self):
        np.random.seed(12345)
        self._create_replay_buffer()

        # fill replay buffer with 10 experiences which observation is between 0 and 9
        num_experiences = 10
        self._fill_replay_buffer(num_experiences)

        # make sure that the priority are set to 0 since the buffer is empty
        expected_priority = np.zeros((self._capacity,), dtype=np.float32)
        for i in range(num_experiences):
            if i >= self._capacity:
                break
            expected_priority[i] = 1.0

        # set the loss of numbers larger 5 to be equal to their number
        # set the loss of numbers smaller or equal to 5 close to 0
        indices = [i for i in range(10)]
        priorities = [i if i > 5 else i / 10 for i in range(10)]

        self._replay_buffer.update_prioritized_buffer_priorities(indices, priorities)

        sample_frequency = [0 for _ in range(10)]
        for i in range(1000):
            sample_batch_size = 10
            mini_batch, indices, weights = self._replay_buffer.get_next(prioritized_buffer_beta=0.4,
                                                                        sample_batch_size=sample_batch_size)
            if i % 100 == 0:
                self.validate_data(mini_batch, indices)

            for idx in indices:
                sample_frequency[idx] += 1

        for i in range(10):
            if i <= 5:
                # numbers smaller than 5 should be picked less that 1% of the time
                self.assertLessEqual(sample_frequency[i], 10000 * 5 / 100)
            else:
                # all numbers larger than 5 should be picked between 15% and 25% of the time
                self.assertGreaterEqual(sample_frequency[i], 10000 * 15 / 100)
                self.assertLessEqual(sample_frequency[i], 10000 * 25 / 100)

                # all numbers larger than 5 should be selected more times than the numbers which precedes them and
                # less time than the numbers that follows them
                self.assertGreaterEqual(sample_frequency[i], sample_frequency[i-1])
                if i < 9:
                    self.assertLessEqual(sample_frequency[i], sample_frequency[i+1])

        # set the loss of numbers larger or equal 5 to be close to 0
        # set the loss of numbers smaller to 5 to their number + 5
        indices = [i for i in range(10)]
        priorities = [i/10 if i >= 5 else i + 5 for i in range(10)]

        self._replay_buffer.update_prioritized_buffer_priorities(indices, priorities)

        sample_frequency = [0 for _ in range(10)]
        for i in range(1000):
            sample_batch_size = 10
            mini_batch, indices, weights = self._replay_buffer.get_next(prioritized_buffer_beta=0.4,
                                                                        sample_batch_size=sample_batch_size)
            if i % 100 == 0:
                self.validate_data(mini_batch, indices)

            for idx in indices:
                sample_frequency[idx] += 1

        for i in range(10):
            if i >= 5:
                # numbers larger than 5 should be picked less that 1% of the time
                self.assertLessEqual(sample_frequency[i], 10000 * 5 / 100)
            else:
                # all numbers smaller or equal to 5 should be picked between 12% and 20% of the time
                self.assertGreaterEqual(sample_frequency[i], 10000 * 12 / 100)
                self.assertLessEqual(sample_frequency[i], 10000 * 20 / 100)

                # all numbers smaller or equal to 5 should be selected more times than the numbers which precedes
                # them and less time than the numbers that follows them
                self.assertGreaterEqual(sample_frequency[i], sample_frequency[i - 1])
                if i < 4:
                    self.assertLessEqual(sample_frequency[i], sample_frequency[i + 1])

    def testPrioritizedReplayBufferFull(self):
        np.random.seed(12345)
        capacity = 10
        self._create_replay_buffer(capacity)

        # fill replay buffer with 20 experiences which observation is between 0 and 19. only values from 10 to 19 will
        # remain in the buffer because it's capacity is 10
        num_experiences = 20
        self._fill_replay_buffer(num_experiences)

        # make sure that the priority are set to 1
        expected_priority = np.zeros((self._capacity,), dtype=np.float32)
        for i in range(num_experiences):
            if i >= self._capacity:
                break
            expected_priority[i] = 1.0

        buffer_priority = self._replay_buffer._prioritized_buffer_priorities
        self.assertAllEqual(expected_priority, buffer_priority)

        # set the loss of numbers larger 15 to be equal to their number
        # set the loss of numbers smaller or equal to 15 close to 0
        indices = [i for i in range(10)]
        priorities = [i if i > 5 else i / 10 for i in range(10)]

        self._replay_buffer.update_prioritized_buffer_priorities(indices, priorities)

        sample_frequency = [0 for _ in range(10)]
        for i in range(1000):
            sample_batch_size = 10
            mini_batch, indices, weights = self._replay_buffer.get_next(prioritized_buffer_beta=0.4,
                                                                        sample_batch_size=sample_batch_size)
            if i % 100 == 0:
                self.validate_data(mini_batch, indices+10)

            for idx in indices:
                sample_frequency[idx] += 1

        for i in range(10):
            if i <= 5:
                # numbers smaller than 5 should be picked less that 1% of the time
                self.assertLessEqual(sample_frequency[i], 10000 * 5 / 100)
            else:
                # all numbers larger than 5 should be picked between 15% and 25% of the time
                self.assertGreaterEqual(sample_frequency[i], 10000 * 15 / 100)
                self.assertLessEqual(sample_frequency[i], 10000 * 25 / 100)

                # all numbers larger than 5 should be selected more times than the numbers which precedes them and
                # less time than the numbers that follows them
                self.assertGreaterEqual(sample_frequency[i], sample_frequency[i-1])
                if i < 9:
                    self.assertLessEqual(sample_frequency[i], sample_frequency[i+1])

        # set the loss of numbers larger or equal 5 to be close to 0
        # set the loss of numbers smaller to 5 to their number + 5
        indices = [i for i in range(10)]
        priorities = [i/10 if i >= 5 else i + 5 for i in range(10)]

        self._replay_buffer.update_prioritized_buffer_priorities(indices, priorities)

        sample_frequency = [0 for _ in range(10)]
        for i in range(1000):
            sample_batch_size = 10
            mini_batch, indices, weights = self._replay_buffer.get_next(prioritized_buffer_beta=0.4,
                                                                        sample_batch_size=sample_batch_size)
            if i % 100 == 0:
                self.validate_data(mini_batch, indices + 10)

            for idx in indices:
                sample_frequency[idx] += 1

        for i in range(10):
            if i >= 5:
                # numbers larger than 5 should be picked less that 1% of the time
                self.assertLessEqual(sample_frequency[i], 10000 * 5 / 100)
            else:
                # all numbers smaller or equal to 5 should be picked between 12% and 20% of the time
                self.assertGreaterEqual(sample_frequency[i], 10000 * 12 / 100)
                self.assertLessEqual(sample_frequency[i], 10000 * 20 / 100)

                # all numbers smaller or equal to 5 should be selected more times than the numbers which precedes
                # them and less time than the numbers that follows them
                self.assertGreaterEqual(sample_frequency[i], sample_frequency[i - 1])
                if i < 4:
                    self.assertLessEqual(sample_frequency[i], sample_frequency[i + 1])
