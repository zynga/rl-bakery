from tensorflow.python.data.util import nest as data_nest
from tf_agents.replay_buffers import table
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer, _valid_range_ids

import collections
import logging
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)

# this is set so that mathematical operations on NaN are raised.
np.seterr(all='raise')

# NOTE: storing the weights in 'probabilities' to be consistent with uniform buffer
BufferInfo = collections.namedtuple('BufferInfo', ['ids', 'probabilities'])

MIN_PRIORITY = 0.0001


class TfPrioritizedReplayBuffer(TFUniformReplayBuffer):
    """
    This class implements a prioritized replay buffer based on the paper linked below.
    Link to Paper: https://arxiv.org/pdf/1511.05952.pdf

    This replay buffer will track priorities, sample a batch according to them, calculate weights and let us
    update priorities after the loss has become known.

    For simplicity, sampling the experiences is implemented in the O(N). For more efficient way to perform sampling,
    the use of trees is required: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py#L71

    This priority buffer is designed to only handle 1 step mini_batch extractions. Time stacking is therefore not
    supported either.

    This priority buffer is also design to only 1 max_length(Segments of batches).
    This simplifies the way the priorities are tracked. Our current use case requires adding a batch once after
    initializing a buffer which makes it possible to use max_length == 1.
    """

    def __init__(self, data_spec,
                 batch_size,
                 max_length=1,
                 alpha=0.6,
                 scope='TFUniformReplayBuffer',
                 device='cpu:*',
                 table_fn=table.Table):
        """
        Args:
            data_spec: A TensorSpec or a list/tuple/nest of TensorSpecs describing a
                       single item that can be stored in this buffer.
            batch_size: Batch dimension of tensors when adding to buffer.
            max_length: The maximum number of items that can be stored in a single
                        batch segment of the buffer.
            alpha: α determines how much prioritization is used, with α = 0 corresponding to the uniform case.
            scope: Scope prefix for variables and ops created by this class.
            device: A TensorFlow device to place the Variables and ops.
            table_fn: Function to create tables `table_fn(data_spec, capacity)` that
                      can read/write nested tensors.
        Raises:
            ValueError: If batch_size does not evenly divide capacity.
        """
        if max_length != 1:
            raise NotImplementedError('TfPrioritizedReplayBuffer only supports a batch segments '
                                      'size of 1, but received `max_length` {}.'.format(max_length))

        super(TfPrioritizedReplayBuffer, self).__init__(data_spec, batch_size, max_length, scope, device, table_fn)
        logger.info("Creating an instance of %s. Params: data_spec: %s, capacity: %s, alpha: %s, segment_length: %s" %
                    (str(type(self).__name__), str(data_spec), str(batch_size), str(alpha), str(max_length)))

        self._alpha = alpha

        # an array in which we keep track of the priorities. The size of this array is equal to the size of the replay
        # buffer. Items stored at a given index in the Priority array map to the experience at the same index in the
        # buffer. The content of the items in the priority array represent the loss of their respective experience the
        # last time that experience was used for training.
        self._priority_table = np.zeros((self._capacity_value,), dtype=np.float32)

    def _add_batch(self, items):
        """Adds a batch of items to the replay buffer.

        Args:
          items: A tensor or list/tuple/nest of tensors representing a batch of
          items to be added to the replay buffer. Each element of `items` must match
          the data_spec of this class. Should be shape [batch_size, data_spec, ...]
        Returns:
          An op that adds `items` to the replay buffer.
        Raises:
          ValueError: If called more than once.
        """
        super()._add_batch(items)
        logger.info("Adding a batch of %s experiences to Replay buffer" % str(self._capacity_value))
        # WARNING: This is implemented under the assumption that add batch is only called once
        self._priority_table = np.ones(self._capacity_value, dtype=np.float32)

    @staticmethod
    def compute_beta(beta, curr_iter, total_iter):
        return min(1.0, beta + curr_iter * (1.0 - beta) / total_iter)

    def get_next(self,
                 sample_batch_size=None,
                 beta=0.4,
                 num_steps=1,
                 time_stacked=False):
        """Returns an item or batch of items sampled based on the experience's priorities from the buffer.

            Only num_steps = 1 and time_stacked=False are supported.

            Args:
                sample_batch_size: (Optional.) An optional batch_size to specify the
                    number of items to return. See get_next() documentation.
                beta (float): ratio to use when computing the importance sampling weights
                num_steps: (Optional.)  Optional way to specify that sub-episodes are
                    desired. See get_next() documentation.
                time_stacked: Bool, when true and num_steps > 1 get_next on the buffer
                    would return the items stack on the time dimension. The outputs would be
                    [B, T, ..] if sample_batch_size is given or [T, ..] otherwise.
            Returns:
                A 2 tuple, containing:
                    - An item, sequence of items, or batch thereof sampled uniformly
                    from the buffer.
                    - BufferInfo NamedTuple, containing:
                        - The items' ids.
                        - The importance sampling weight of each item.
            Raises:
                ValueError: if num_steps is bigger than the capacity.
        """
        num_steps_value = num_steps if num_steps is not None else 1

        if num_steps_value != 1:
            raise NotImplementedError('TfPrioritizedReplayBuffer only supports a batches with num_step '
                                      'size of 1, but received batch with num_steps'
                                      'size {}.'.format(num_steps_value))
        with tf.device(self._device), tf.name_scope(self._scope):
            with tf.name_scope('get_next'):
                min_val, max_val = _valid_range_ids(
                    self._get_last_id(), self._max_length, num_steps)
                tf.compat.v1.assert_greater(
                    max_val,
                    min_val,
                    message='TFUniformReplayBuffer is empty. Make sure to add items '
                            'before sampling the buffer.')

        # select index based on the priorities of the experiences in the buffer
        selected_idx, selected_weight = self._sample_ids(sample_batch_size, beta)

        # convert ids to tensor
        rows_to_get = tf.convert_to_tensor(selected_idx)
        importance_sampling_weigths = tf.convert_to_tensor(selected_weight)
        data = self._data_table.read(rows_to_get)

        buffer_info = BufferInfo(ids=rows_to_get, probabilities=importance_sampling_weigths)
        return data, buffer_info

    def _sample_ids(self, sample_size, beta):
        """
        select the indices of the experience to pull from the prioritize replay buffer in addition to compute it's
        importance sampling weight.

        Params:
            sample_size (int): number of ids to retrieve
            beta (float): This a factor to which the Important Sampling is present
        """
        priorities = self._priority_table

        # update the priorities by the power of alpha
        probabilities = priorities ** self._alpha
        probabilities /= probabilities.sum()

        # select an index stochastically
        idx = np.random.choice(len(probabilities), sample_size, p=probabilities)

        # compute important sampling weights of the selected indices
        selected_probabilities = probabilities[idx]
        importance_sampling_weights = (self._capacity_value * selected_probabilities) ** (-beta)
        importance_sampling_weights /= importance_sampling_weights.max()

        return idx, importance_sampling_weights

    def update_priorities(self, indices, new_priorities):
        """
        Updates new priorities of the experiences in the given indices.

        Params:
            batch_indices: the indices to update
            batch_priorities: the new priorities
        """
        for idx, priority in zip(indices, new_priorities):
            ceil_priority = max(priority, MIN_PRIORITY)
            self._priority_table[idx] = ceil_priority

    def as_dataset(self,
                   sample_batch_size=None,
                   beta=0.4,
                   num_steps=None,
                   num_parallel_calls=None):
        """Creates a dataset that returns entries from the buffer.

            Args:
                sample_batch_size: (Optional.) An optional batch_size to specify the
                    number of items to return. See as_dataset() documentation.
                beta: This a factor to which the Important Sampling is present
                num_steps: (Optional.)  Optional way to specify that sub-episodes are
                    desired. See as_dataset() documentation.
                num_parallel_calls: (Optional.) Number elements to process in parallel.
                See as_dataset() documentation.
            Returns:
                A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
                    - An item or sequence of items or batch thereof
                    - Auxiliary info for the items (i.e. ids, probs).
            Raises:
                ValueError: If the data spec contains lists that must be converted to
                tuples.
        """
        # data_tf.nest.flatten does not flatten python lists, nest.flatten does.
        if tf.nest.flatten(self._data_spec) != data_nest.flatten(self._data_spec):
            raise ValueError(
                'Cannot perform gather; data spec contains lists and this conflicts '
                'with gathering operator.  Convert any lists to tuples.  '
                'For example, if your spec looks like [a, b, c], '
                'change it to (a, b, c).  Spec structure is:\n  {}'.format(
                    tf.nest.map_structure(lambda spec: spec.dtype, self._data_spec)))

        def get_next(_):
            return self.get_next(sample_batch_size, beta=beta, num_steps=num_steps, time_stacked=False)

        return tf.data.experimental.Counter().map(
            get_next,
            num_parallel_calls=num_parallel_calls)
