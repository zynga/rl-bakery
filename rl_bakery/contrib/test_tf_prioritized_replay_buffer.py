#
# Based on work found at:
# https://github.com/tensorflow/agents/blob/master/tf_agents/replay_buffers/tf_uniform_replay_buffer_test.py
#

"""Tests for tf_prioritized_replay_buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from rl_bakery.contrib.tf_prioritized_replay_buffer import TfPrioritizedReplayBuffer

from tf_agents import specs
from tf_agents.utils import common
from tf_agents.utils import test_utils


def _get_add_op(spec, replay_buffer, batch_size):
    # TODO(b/68398658) Remove dtypes once scatter_update is fixed.
    action = tf.constant(1 * np.ones(spec[0].shape.as_list(), dtype=np.float32))
    lidar = tf.constant(2 * np.ones(spec[1][0].shape.as_list(), dtype=np.float32))
    camera = tf.constant(
        3 * np.ones(spec[1][1].shape.as_list(), dtype=np.float32))
    values = [action, [lidar, camera]]
    values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * batch_size),
                                           values)

    return values, replay_buffer.add_batch(values_batched)


class TFPrioritizedReplayBufferTest(parameterized.TestCase, tf.test.TestCase):

    def _assert_contains(self, list1, list2):
        self.assertTrue(
            test_utils.contains(list1, list2), '%s vs. %s' % (list1, list2))

    def _assert_circular_ordering(self, expected_order, given_order):
        for i in range(len(given_order)):
            self.assertIn(given_order[i], expected_order)
            if i > 0:
                prev_idx = expected_order.index(given_order[i - 1])
                cur_idx = expected_order.index(given_order[i])
                self.assertEqual(cur_idx, (prev_idx + 1) % len(expected_order))

    def _data_spec(self):
        return [
            specs.TensorSpec([3], tf.float32, 'action'),
            [
                specs.TensorSpec([5], tf.float32, 'lidar'),
                specs.TensorSpec([3, 2], tf.float32, 'camera')
            ]
        ]

    def test_add_batch_one(self):
        batch_size = 1
        spec = self._data_spec()
        replay_buffer = TfPrioritizedReplayBuffer(
            spec,
            batch_size=batch_size,
            max_length=1,
            scope='rb{}'.format(batch_size))

        values, add_op = _get_add_op(spec, replay_buffer, batch_size)
        sample, _ = replay_buffer.get_next()

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(add_op)
        sample_ = self.evaluate(sample)
        values_ = self.evaluate(values)
        tf.nest.map_structure(self.assertAllClose, values_, sample_)

    def test_add_batch_five(self):
        batch_size = 5
        spec = self._data_spec()
        replay_buffer = TfPrioritizedReplayBuffer(
            spec,
            batch_size=batch_size,
            max_length=1,
            scope='rb{}'.format(batch_size))

        values, add_op = _get_add_op(spec, replay_buffer, batch_size)
        sample, _ = replay_buffer.get_next()

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(add_op)
        sample_ = self.evaluate(sample)
        values_ = self.evaluate(values)
        tf.nest.map_structure(self.assertAllClose, values_, sample_)

    def test_get_next_empty(self):
        spec = self._data_spec()
        replay_buffer = TfPrioritizedReplayBuffer(
            spec, batch_size=1, max_length=1)

        with self.assertRaisesRegexp(
                tf.errors.InvalidArgumentError, 'TFUniformReplayBuffer is empty. Make '
                                                'sure to add items before sampling the buffer.'):
            self.evaluate(tf.compat.v1.global_variables_initializer())
            sample, _ = replay_buffer.get_next()
            self.evaluate(sample)

    def test_add_single_sample_batch(self):
        batch_size = 1
        spec = self._data_spec()
        replay_buffer = TfPrioritizedReplayBuffer(
            spec, batch_size=batch_size, max_length=1)

        values, add_op = _get_add_op(spec, replay_buffer, batch_size)
        sample, _ = replay_buffer.get_next(sample_batch_size=3)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(add_op)
        values_ = self.evaluate(values)
        sample_ = self.evaluate(sample)
        tf.nest.map_structure(lambda x, y: self._assert_contains([x], list(y)),
                              values_, sample_)

    def test_clear(self):
        batch_size = 1
        spec = self._data_spec()
        replay_buffer = TfPrioritizedReplayBuffer(
            spec, batch_size=batch_size, max_length=1)

        self.evaluate(tf.compat.v1.global_variables_initializer())

        initial_id = self.evaluate(replay_buffer._get_last_id())
        empty_items = self.evaluate(replay_buffer.gather_all())

        values, _ = self.evaluate(_get_add_op(spec, replay_buffer, batch_size))
        sample, _ = self.evaluate(replay_buffer.get_next(sample_batch_size=3))
        tf.nest.map_structure(lambda x, y: self._assert_contains([x], list(y)),
                              values, sample)
        self.assertNotEqual(initial_id, self.evaluate(replay_buffer._get_last_id()))

        self.evaluate(replay_buffer.clear())
        self.assertEqual(initial_id, self.evaluate(replay_buffer._get_last_id()))

        def check_np_arrays_everything_equal(x, y):
            np.testing.assert_equal(x, y)
            self.assertEqual(x.dtype, y.dtype)

        tf.nest.map_structure(check_np_arrays_everything_equal, empty_items,
                              self.evaluate(replay_buffer.gather_all()))

    def test_clear_all_variables(self):
        batch_size = 1
        spec = self._data_spec()
        replay_buffer = TfPrioritizedReplayBuffer(
            spec, batch_size=batch_size, max_length=1)

        action = tf.constant(1 * np.ones(spec[0].shape.as_list(), dtype=np.float32))
        lidar = tf.constant(
            2 * np.ones(spec[1][0].shape.as_list(), dtype=np.float32))
        camera = tf.constant(
            3 * np.ones(spec[1][1].shape.as_list(), dtype=np.float32))
        values = [action, [lidar, camera]]
        values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * batch_size),
                                               values)

        if tf.executing_eagerly():
            def add_batch():
                return replay_buffer.add_batch(values_batched)

            add_op = add_batch
        else:
            add_op = replay_buffer.add_batch(values_batched)

        def get_table_vars():
            return [var for var in replay_buffer.variables() if 'Table' in var.name]

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(replay_buffer._clear(clear_all_variables=True))
        empty_table_vars = self.evaluate(get_table_vars())
        initial_id = self.evaluate(replay_buffer._get_last_id())
        empty_items = self.evaluate(replay_buffer.gather_all())
        self.evaluate(add_op)
        self.evaluate(add_op)
        self.evaluate(add_op)
        self.evaluate(add_op)
        values_ = self.evaluate(values)
        sample, _ = self.evaluate(replay_buffer.get_next(sample_batch_size=3))
        tf.nest.map_structure(lambda x, y: self._assert_contains([x], list(y)),
                              values_, sample)
        self.assertNotEqual(initial_id, self.evaluate(replay_buffer._get_last_id()))

        tf.nest.map_structure(lambda x, y: self.assertFalse(np.all(x == y)),
                              empty_table_vars, self.evaluate(get_table_vars()))

        self.evaluate(replay_buffer._clear(clear_all_variables=True))
        self.assertEqual(initial_id, self.evaluate(replay_buffer._get_last_id()))

        def check_np_arrays_everything_equal(x, y):
            np.testing.assert_equal(x, y)
            self.assertEqual(x.dtype, y.dtype)

        tf.nest.map_structure(check_np_arrays_everything_equal, empty_items,
                              self.evaluate(replay_buffer.gather_all()))

    def test_gather_all_atch_one(self):
        batch_size = 1
        spec = specs.TensorSpec([], tf.int64, 'action')
        replay_buffer = TfPrioritizedReplayBuffer(spec, batch_size=batch_size)

        @common.function(autograph=True)
        def add_data():
            batch = tf.range(0, batch_size, 1, dtype=tf.int64)
            replay_buffer.add_batch(batch)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(add_data())

        items = replay_buffer.gather_all()
        expected = [list(range(i, i + 1)) for i in range(0, batch_size)]

        items_ = self.evaluate(items)
        self.assertAllClose(expected, items_)

    def test_gather_all_batch_five(self):
        batch_size = 5
        spec = specs.TensorSpec([], tf.int64, 'action')
        replay_buffer = TfPrioritizedReplayBuffer(
            spec, batch_size=batch_size)

        @common.function(autograph=True)
        def add_data():
            batch = tf.range(0, batch_size, 1, dtype=tf.int64)
            replay_buffer.add_batch(batch)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(add_data())

        items = replay_buffer.gather_all()
        expected = [list(range(i, i + 1)) for i in range(0, batch_size)]

        items_ = self.evaluate(items)
        self.assertAllClose(expected, items_)

    def test_gather_allEmpty_batch_one(self):
        batch_size = 1
        spec = specs.TensorSpec([], tf.int32, 'action')
        replay_buffer = TfPrioritizedReplayBuffer(
            spec, batch_size=batch_size)

        items = replay_buffer.gather_all()
        expected = [[]] * batch_size

        self.evaluate(tf.compat.v1.global_variables_initializer())
        items_ = self.evaluate(items)
        self.assertAllClose(expected, items_)

    def test_gather_all_empty_batch_five(self):
        batch_size = 5
        spec = specs.TensorSpec([], tf.int32, 'action')
        replay_buffer = TfPrioritizedReplayBuffer(
            spec, batch_size=batch_size)

        items = replay_buffer.gather_all()
        expected = [[]] * batch_size

        self.evaluate(tf.compat.v1.global_variables_initializer())
        items_ = self.evaluate(items)
        self.assertAllClose(expected, items_)

    def test_sample_batch_Correct_probabilities_batch_ten(self):
        buffer_batch_size = 10
        alpha = 0.6
        spec = specs.TensorSpec([], tf.int32, 'action')
        replay_buffer = TfPrioritizedReplayBuffer(spec, batch_size=buffer_batch_size, max_length=1, alpha=alpha)

        experience = []
        experience_shape = (1,)
        for k in range(buffer_batch_size):
            experience.append(np.full(experience_shape, k, dtype=np.int32))

        tf_experience = tf.convert_to_tensor(experience)
        replay_buffer.add_batch(tf_experience)

        sample_batch_size = 2

        @common.function
        def probabilities():
            _, buffer_info = replay_buffer.get_next(sample_batch_size=sample_batch_size)
            return buffer_info.probabilities

        self.evaluate(tf.compat.v1.global_variables_initializer())

        beta = 0.6
        selected_probabilities = 1 ** alpha / ((1 ** alpha) * buffer_batch_size)
        expected_weights = [(buffer_batch_size * selected_probabilities) ** (-beta) for _ in range(sample_batch_size)]

        weights = self.evaluate(probabilities())
        self.assertAllClose(expected_weights, weights)

    def test_sample_batch_correct_probabilities_batch_ten_as_dataset(self):
        buffer_batch_size = 10
        alpha = 0.6
        spec = specs.TensorSpec([], tf.int32, 'action')
        replay_buffer = TfPrioritizedReplayBuffer(spec, batch_size=buffer_batch_size, max_length=1, alpha=alpha)

        experience = []
        experience_shape = (1,)
        for k in range(buffer_batch_size):
            experience.append(np.full(experience_shape, k, dtype=np.int32))

        tf_experience = tf.convert_to_tensor(experience)
        replay_buffer.add_batch(tf_experience)

        sample_batch_size = 2

        self.evaluate(tf.compat.v1.global_variables_initializer())
        beta = 0.6
        ds = replay_buffer.as_dataset(sample_batch_size=sample_batch_size, beta=beta)

        itr = iter(ds)

        def next_iter():
            return next(itr)

        sample = next_iter

        selected_probabilities = 1 ** alpha / ((1 ** alpha) * buffer_batch_size)
        expected_weights = [(buffer_batch_size * selected_probabilities) ** (-beta) for _ in range(sample_batch_size)]
        res = self.evaluate(sample)
        weights = res[1].probabilities
        self.assertAllClose(expected_weights, weights)

    def validate_data(self, mini_batch, indices):
        for idx, item in enumerate(mini_batch):
            expected_item = indices[idx]
            self.assertAllEqual(item, expected_item)

    def test_prioritized_replay_buffer(self):
        np.random.seed(123)

        buffer_batch_size = 10
        alpha = 0.6
        spec = specs.TensorSpec([], tf.int32, 'action')
        replay_buffer = TfPrioritizedReplayBuffer(spec, batch_size=buffer_batch_size, max_length=1, alpha=alpha)

        # make sure that the priority are set to 0 since the buffer is empty
        expected_priority = np.zeros((buffer_batch_size,), dtype=np.float32)
        for i in range(buffer_batch_size):
            if i >= buffer_batch_size:
                break
            expected_priority[i] = 1.0

        experience = []
        experience_shape = (1,)
        for k in range(buffer_batch_size):
            experience.append(np.full(experience_shape, k, dtype=np.int32))

        tf_experience = tf.convert_to_tensor(experience)
        replay_buffer.add_batch(tf_experience)

        sample_frequency = [0 for _ in range(10)]
        for i in range(1000):
            sample_batch_size = 10
            mini_batch, metadata = replay_buffer.get_next(beta=0.4, sample_batch_size=sample_batch_size)
            indices_tf = metadata.ids
            # indices = self.evaluate(indices_tf)
            indices = indices_tf.numpy()
            if i % 100 == 0:
                self.validate_data(mini_batch, indices)

            for idx in indices:
                sample_frequency[idx] += 1

            # set the loss of numbers larger 5 to be equal to their number
            # set the loss of numbers smaller or equal to 5 close to 0

            priorities = [i if i > 5 else i / 10 for i in indices]

            replay_buffer.update_priorities(indices, priorities)

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

        indices = [i for i in range(10)]
        priorities = [1 for _ in range(10)]

        replay_buffer.update_priorities(indices, priorities)

        # set the loss of numbers larger or equal 5 to be close to 0
        # set the loss of numbers smaller to 5 to their number + 5
        sample_frequency = [0 for _ in range(10)]
        for i in range(1000):
            sample_batch_size = 10
            mini_batch, metadata = replay_buffer.get_next(beta=0.4, sample_batch_size=sample_batch_size)
            indices_tf = metadata.ids

            indices = indices_tf.numpy()
            if i % 100 == 0:
                self.validate_data(mini_batch, indices)

            for idx in indices:
                sample_frequency[idx] += 1

            # set the loss of numbers larger 5 to be equal to their number
            # set the loss of numbers smaller or equal to 5 close to 0

            priorities = [i / 10 if i >= 5 else i + 5 for i in indices]
            replay_buffer.update_priorities(indices, priorities)

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

    def test_prioritized_replay_buffer_as_dataset(self):
        np.random.seed(123)

        buffer_batch_size = 10
        alpha = 0.6
        spec = specs.TensorSpec([], tf.int32, 'action')
        replay_buffer = TfPrioritizedReplayBuffer(spec, batch_size=buffer_batch_size, max_length=1, alpha=alpha)

        # make sure that the priority are set to 0 since the buffer is empty
        expected_priority = np.zeros((buffer_batch_size,), dtype=np.float32)
        for i in range(buffer_batch_size):
            if i >= buffer_batch_size:
                break
            expected_priority[i] = 1.0

        experience = []
        experience_shape = (1,)
        for k in range(buffer_batch_size):
            experience.append(np.full(experience_shape, k, dtype=np.int32))

        tf_experience = tf.convert_to_tensor(experience)
        replay_buffer.add_batch(tf_experience)

        sample_batch_size = 10
        beta = 0.4

        sample_frequency = [0 for _ in range(10)]
        for i in range(15*3):
            ds = replay_buffer.as_dataset(sample_batch_size=sample_batch_size, beta=beta)
            itr = iter(ds)
            for j in range(int(100/3)):
                mini_batch, metadata = next(itr)
                indices_tf = metadata.ids
                # indices = self.evaluate(indices_tf)
                indices = indices_tf.numpy()
                if i % 100 == 0:
                    self.validate_data(mini_batch, indices)

                for idx in indices:
                    sample_frequency[idx] += 1

                # set the loss of numbers larger 5 to be equal to their number
                # set the loss of numbers smaller or equal to 5 close to 0

                priorities = [i if i > 5 else i / 10 for i in indices]

                replay_buffer.update_priorities(indices, priorities)

        for i in range(10):
            if i <= 5:
                # numbers smaller than 5 should be picked less that 1% of the time
                self.assertLessEqual(sample_frequency[i], 15000 * 5 / 100)
            else:
                # all numbers larger than 5 should be picked between 15% and 25% of the time
                self.assertGreaterEqual(sample_frequency[i], 15000 * 15 / 100)
                self.assertLessEqual(sample_frequency[i], 15000 * 30 / 100)

                # all numbers larger than 5 should be selected more times than the numbers which precedes them and
                # less time than the numbers that follows them
                self.assertGreaterEqual(sample_frequency[i], sample_frequency[i-1])
                if i < 9:
                    self.assertLessEqual(sample_frequency[i], sample_frequency[i+1])

        indices = [i for i in range(10)]
        priorities = [1 for _ in range(10)]

        replay_buffer.update_priorities(indices, priorities)
        np.random.seed(12323423)
        # set the loss of numbers larger or equal 5 to be close to 0
        # set the loss of numbers smaller to 5 to their number + 5
        sample_frequency = [0 for _ in range(10)]
        for i in range(15*20):
            ds = replay_buffer.as_dataset(sample_batch_size=sample_batch_size, beta=beta)
            itr = iter(ds)
            for j in range(int(100/20)):
                mini_batch, metadata = next(itr)
                indices_tf = metadata.ids

                indices = indices_tf.numpy()
                if i % 100 == 0:
                    self.validate_data(mini_batch, indices)

                for idx in indices:
                    sample_frequency[idx] += 1

                # set the loss of numbers larger 5 to be equal to their number
                # set the loss of numbers smaller or equal to 5 close to 0

                priorities = [i / 10 if i >= 5 else i + 5 for i in indices]
                replay_buffer.update_priorities(indices, priorities)

        for i in range(10):
            if i >= 5:
                # numbers larger than 5 should be picked less that 1% of the time
                self.assertLessEqual(sample_frequency[i], 15000 * 5 / 100)
            else:
                # all numbers smaller or equal to 5 should be picked between 12% and 20% of the time
                self.assertGreaterEqual(sample_frequency[i], 15000 * 10 / 100)
                self.assertLessEqual(sample_frequency[i], 15000 * 25 / 100)

                # all numbers smaller or equal to 5 should be selected more times than the numbers which precedes
                # them and less time than the numbers that follows them
                self.assertGreaterEqual(sample_frequency[i], sample_frequency[i - 1])
                if i < 4:
                    self.assertLessEqual(sample_frequency[i], sample_frequency[i + 1])


if __name__ == '__main__':
    tf.test.main()
