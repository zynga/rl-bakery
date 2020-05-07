from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.specs import array_spec
from tf_agents.utils import nest_utils
from tf_agents.utils import numpy_storage

import logging
import numpy as np
import tensorflow as tf
import threading

logger = logging.getLogger(__name__)


class PyPrioritizedReplayBuffer(ReplayBuffer):
    """
    This class implements a prioritized replay buffer based on the paper linked below.
    Link to Paper: https://arxiv.org/pdf/1511.05952.pdf

    This replay buffer will track priorities, sample a batch according to them, calculate weights and let us
    update priorities after the loss has become known.

    For simplicity, sampling the experiences is implemented in the O(N). For more efficient way to perform sampling,
    the use of trees is required: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py#L71
    """

    def __init__(self, data_spec, capacity, alpha=0.6):
        """
        Params:
            data_spec: An ArraySpec or a list/tuple/nest of ArraySpecs describing a single item that can be stored
                       in this buffer.
            capacity: The maximum number of items that can be stored in the buffer.
            alpha: α determines how much prioritization is used, with α = 0 corresponding to the uniform case.
        """
        super(PyPrioritizedReplayBuffer, self).__init__(data_spec, capacity)
        logger.info("Creating an instance of %s. Params: data_spec: %s, capacity: %s, alpha: %s" %
                    (str(type(self).__name__), str(data_spec), str(capacity), str(alpha)))

        # State variables needed to maintain the replay buffer. These were copied from the uniform replay buffer
        self._storage = numpy_storage.NumpyStorage(self._encoded_data_spec(), capacity)
        self._lock = threading.Lock()
        self._np_state = numpy_storage.NumpyState()

        # Adding elements to the replay buffer is done in a circular way.
        # Keeps track of the actual size of the replay buffer and the location
        # where to add new elements.
        self._np_state.size = np.int64(0)
        self._np_state.cur_id = np.int64(0)

        # Total number of items that went through the replay buffer.
        self._np_state.item_count = np.int64(0)

        self._prioritized_buffer_alpha = alpha
        self._prioritized_buffer_capacity = capacity

        # an array in which we keep track of the priorities. The size of this array is equal to the size of the replay
        # buffer. Items stored at a given index in the Priority array map to the experience at the same index in the
        # buffer. The content of the items in the priority array represent the loss of their respective experience the
        # last time that experience was used for training.
        self._prioritized_buffer_priorities = np.zeros((capacity,), dtype=np.float32)

    def _encoded_data_spec(self):
        """Spec of data items after encoding using _encode."""
        return self._data_spec

    def _encode(self, item):
        """Encodes an item (before adding it to the buffer)."""
        return item

    def _decode(self, item):
        """Decodes an item."""
        return item

    def _on_delete(self, encoded_item):
        """Do any necessary cleanup."""
        pass

    @property
    def size(self):
        return self._np_state.size

    def _add_batch(self, items):
        """
        Add the experiences in the batch to the replay buffer. Only batches of size 1 are supported at the moment

        Params:
            items: this contains the experiences to be added
        """
        logger.info("Adding a batch of 1 experiences to Replay buffer")

        outer_shape = nest_utils.get_outer_array_shape(items, self._data_spec)
        if outer_shape[0] != 1:
            raise NotImplementedError('PyPrioritizedReplayBuffer only supports a batch '
                                      'size of 1, but received `items` with batch '
                                      'size {}.'.format(outer_shape[0]))

        item = nest_utils.unbatch_nested_array(items)

        # get maximum priority in the replay buffer or set it's initial value is 1
        max_priority = self._prioritized_buffer_priorities.max() if self._np_state.size > 0 else 1.0

        with self._lock:
            if self._np_state.size == self._prioritized_buffer_capacity:
                # If we are at capacity, we are deleting element cur_id.
                self._on_delete(self._storage.get(self._np_state.cur_id))

            self._storage.set(self._np_state.cur_id, self._encode(item))
            # add the max priority of the experience to the priority array
            self._prioritized_buffer_priorities[self._np_state.cur_id] = max_priority

            self._np_state.size = np.minimum(self._np_state.size + 1, self._prioritized_buffer_capacity)
            self._np_state.cur_id = (self._np_state.cur_id + 1) % self._prioritized_buffer_capacity
            self._np_state.item_count += 1

    def get_next(self, sample_batch_size=None, prioritized_buffer_beta=0.4, num_steps=None, time_stacked=True):
        """
        Build next batch of experiences while computing the importance sampling weights of the selected experiences

        Params:
            sample_batch_size (int): batch size
            beta (float): ratio to use when computing the importance sampling weights
            num_steps (int): number of steps to load. Only 1 is supported at the moment
            time_stacked (bool): whether the timesteps are stacked or not.

        Returns:
            (Trajectory): mini batch of experiences
            (int list): the indices of the selected experiences in the replay buffer
            (float list): importance sampling weights to use when training using the experiences.
        """
        num_steps_value = num_steps if num_steps is not None else 1

        if num_steps_value != 1:
            raise NotImplementedError('PyPrioritizedReplayBuffer only supports a batches with num_step '
                                      'size of 1, but received batch with num_steps'
                                      'size {}.'.format(num_steps_value))

        def get_single_experience(b):
            """Gets a single experience from the replay buffer."""

            with self._lock:
                # return empty items if the buffer is empty
                if self._np_state.size <= 0:
                    def empty_item(spec):
                        return np.empty(spec.shape, dtype=spec.dtype)

                    item = tf.nest.map_structure(empty_item, self.data_spec)
                    selected_idx = -1
                    selected_weight = -1

                    return item, selected_idx, selected_weight

                # select index based on the priorities of the experiences in the buffer
                selected_idx, selected_weight = self.select_prioritized_experience(b)

                # get item
                item = self._decode(self._storage.get(selected_idx % self._prioritized_buffer_capacity))

            return item, selected_idx, selected_weight

        if sample_batch_size is None:
            return get_single_experience(prioritized_buffer_beta)
        else:
            experiences = []
            buffer_indices = []
            importance_sampling_weights = []
            for _ in range(sample_batch_size):
                experience, idx, weight = get_single_experience(prioritized_buffer_beta)
                experiences.append(experience)
                buffer_indices.append(idx)
                importance_sampling_weights.append(weight)

            buffer_indices = np.array(buffer_indices)
            importance_sampling_weights = np.array(importance_sampling_weights, dtype=np.float32)

            # normalize weight
            importance_sampling_weights = np.divide(importance_sampling_weights, importance_sampling_weights.max())

            trajectory = nest_utils.stack_nested_arrays(experiences)
            return trajectory, buffer_indices, importance_sampling_weights

    def select_prioritized_experience(self, prioritized_buffer_beta):
        """
        select the index of the experience to pull from the prioritize replay buffer in addition to compute it's
        importance sampling weight.

        Params:
            prioritized_buffer_beta: This a factor to which the Importance Sampling is present
        """
        # extract priorities
        if self._np_state.size == self._prioritized_buffer_capacity:
            priorities = self._prioritized_buffer_priorities
        else:
            priorities = self._prioritized_buffer_priorities[:self._np_state.size]

        # update the priorities by the power of alpha
        probabilities = priorities ** self._prioritized_buffer_alpha
        probabilities /= probabilities.sum()

        # select an index stochastically
        idx = np.random.choice(self._np_state.size, p=probabilities)

        # compute Importance Sampling weights of the selected indices
        importance_sampling_weight = (self._np_state.size * probabilities[idx]) ** (-prioritized_buffer_beta)
        return idx, importance_sampling_weight

    def update_prioritized_buffer_priorities(self, indices, new_priorities):
        """
        Updates new priorities of the experiences in the given indices.

        Params:
            batch_indices: the indices to update
            batch_priorities: the new priorities
        """
        for idx, priority in zip(indices, new_priorities):
            self._prioritized_buffer_priorities[idx] = priority

    def as_dataset(self, sample_batch_size=None, prioritized_buffer_beta=0.4, num_steps=None, num_parallel_calls=None):
        """
        build a tf Dataset which will be able to serve batches of experiences at scale

        Params:
            sample_batch_size: size of the batches it will return
            prioritized_buffer_beta: This a factor to which the Importance Sampling is present
            num_steps (int): number of steps to load. Only 1 is supported at the moment
            num_parallel_calls: number of calls to perform in parallel.

        Returns:
            tf dataset
        """
        if num_parallel_calls is not None:
            raise NotImplementedError('PyUniformReplayBuffer does not support num_parallel_calls (must be None).')

        data_spec = self._data_spec

        num_steps_value = num_steps if num_steps is not None else 1

        if num_steps_value != 1:
            raise NotImplementedError('PyPrioritizedReplayBuffer only supports a batches with num_step '
                                      'size of 1, but received batch with num_steps'
                                      'size {}.'.format(num_steps_value))

        if sample_batch_size is not None:
            data_spec = array_spec.add_outer_dims_nest(data_spec, (sample_batch_size,))

        experiences_shapes = tuple(s.shape for s in tf.nest.flatten(data_spec))
        experiences_dtypes = tuple(s.dtype for s in tf.nest.flatten(data_spec))

        indices_shape = (sample_batch_size,) if sample_batch_size else ()
        indices_dtype = np.int32

        weight_shape = (sample_batch_size,) if sample_batch_size else ()
        weight_dtypes = np.float32

        shapes = {"experiences": experiences_shapes, "indices": indices_shape, "weights": weight_shape}
        dtypes = {"experiences": experiences_dtypes, "indices": indices_dtype, "weights": weight_dtypes}

        def generator_fn():
            while True:
                if sample_batch_size is not None:
                    batch = [self.get_next(num_steps=num_steps_value, time_stacked=False,
                                           prioritized_buffer_beta=prioritized_buffer_beta)
                             for _ in range(sample_batch_size)]
                    item, item_idx, item_weight = nest_utils.stack_nested_arrays(batch)
                else:
                    item, item_idx, item_weight = self.get_next(num_steps=num_steps_value, time_stacked=False,
                                                                prioritized_buffer_beta=prioritized_buffer_beta)

                yield {"experiences": tuple(tf.nest.flatten(item)), "indices": item_idx, "weights": item_weight}

        def pack_items(*items):
            if len(items) > 0:
                experience = tf.nest.pack_sequence_as(data_spec, items[0]["experiences"])
                return experience, items[0]["indices"], items[0]["weights"]
            else:
                raise Exception("No items to pack")

        ds = tf.data.Dataset.from_generator(generator_fn, dtypes, shapes).map(pack_items)

        return ds

    def _gather_all(self):
        data = [self._decode(self._storage.get(idx))
                for idx in range(self._prioritized_buffer_capacity)]
        stacked = nest_utils.stack_nested_arrays(data)
        batched = tf.nest.map_structure(lambda t: np.expand_dims(t, 0), stacked)
        return batched

    def _clear(self):
        self._np_state.size = np.int64(0)
        self._np_state.cur_id = np.int64(0)
