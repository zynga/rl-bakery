from rl_bakery.replay_buffer.tf_replay_buffer_abstract import TFReplayBufferAbstract, build_tf_trajectory
from rl_bakery.contrib.tf_prioritized_replay_buffer import TfPrioritizedReplayBuffer as TFReplayBuffer

import tensorflow as tf

"""
A batched replay buffer of nests of Tensors that samples trajectories based on the 
loss experienced the most recent time the agent trained using them.

Link to paper: https://arxiv.org/pdf/1511.05952.pdf
"""


class TFPrioritizedReplayBuffer(TFReplayBufferAbstract):
    _curr_beta = None

    def __init__(self, collect_data_spec, alpha, beta, training_iterations_num):
        """
        Store replay buffer params

        Params:
            collect_data_spec: spec of the data to be added to the buffer
            alpha: This param is used to determine how much emphasis is given to the priority
            beta:
            training_iterations_num:
        """
        super().__init__(collect_data_spec)
        self._beta = beta
        self._alpha = alpha
        self._training_iterations_total_num = training_iterations_num
        self._metric_tracker = None

    def _init_replay_buffer(self, batch_size, traj_spec):

        buffer_config = {
            "batch_size": batch_size,
            "data_spec": traj_spec,
            "max_length": 1,
            "alpha": self._alpha
        }
        tf.compat.v2.summary.scalar(name="replay_buffer_size", data=batch_size)
        self._replay_buffer = TFReplayBuffer(**buffer_config)

    def add_batch(self, traj_dict):
        """
        add a trajectory to the replay buffer

        Params
            traj (dict[dim]:numpy): a dict of tensors representing the trajectory to be added it to the replay buffer
        """
        collect_spec_dict = self.collect_data_spec._asdict()
        traj_tf, traj_spec = build_tf_trajectory(traj_dict, collect_spec_dict)

        if not self._replay_buffer:
            batch_size = len(traj_dict["observation"])
            self._init_replay_buffer(batch_size, traj_spec)

        self._replay_buffer.add_batch(traj_tf)

    def get_batch(self, batch_size):
        traj, metadata = self._replay_buffer.get_next(sample_batch_size=batch_size, beta=self._curr_beta)

        self._metric_tracker.add_batch_weights(metadata.probabilities)
        self._metric_tracker.add_batch_indices(metadata.ids)

        return traj, metadata

    def pre_process(self, curr_iter):

        if not self._metric_tracker:
            self._metric_tracker = TrainingMetricTracker(self._training_iterations_total_num)

        # compute the beta that will be used when computing the importance sampling weights
        self._curr_beta = self._replay_buffer.compute_beta(self._beta, curr_iter, self._training_iterations_total_num)
        # add important data to the metric tracker
        self._metric_tracker.latest_iteration = curr_iter
        self._metric_tracker.latest_beta = self._curr_beta

    def post_process(self, traj_meta, loss_info, curr_iter):
        indices = traj_meta.ids.numpy()
        # get the loss of every experience using during the training. it is stored in DQNLossInfo
        td_loss = loss_info[1].td_loss.numpy()

        # make sure the td loss array has the same size as the batch
        if td_loss.shape != indices.shape:
            raise Exception("Expected the shape of the loss '%s' to be the same as the shape of the "
                            "indices '%s'" % (str(td_loss.shape), len(indices.shape)))

        # update the prioritized replay buffer
        self._replay_buffer.update_priorities(indices, td_loss)

        self._metric_tracker.log_partial_metrics()
        self._metric_tracker.latest_loss_info = loss_info

        if curr_iter == self._training_iterations_total_num:
            self._metric_tracker.log_summary_metrics()


class TrainingMetricTracker(object):
    _latest_training_progress_percentage = 0
    _max_index = 0
    _sample_frequency = {}
    _weights_summary = None
    _indices_summary = None

    def __init__(self, num_iterations):
        self.num_iterations = num_iterations
        self.latest_loss_info = None
        self.latest_beta = None
        self.latest_batch_weights = None
        self.latest_batch_indices = None
        self.latest_iteration = None

    def add_batch_weights(self, weights_tensor):
        self.latest_batch_weights = weights_tensor
        self._weights_summary = tf.concat([self._weights_summary, weights_tensor], 0) \
            if self._weights_summary is not None else weights_tensor

    def add_batch_indices(self, indices_tensor):
        self.latest_batch_indices = indices_tensor

        self._indices_summary = tf.concat([self._indices_summary, indices_tensor], 0) \
            if self._indices_summary is not None else indices_tensor

        indices = indices_tensor.numpy()
        for idx in indices:
            if idx in self._sample_frequency:
                self._sample_frequency[str(idx)] += 1
            else:
                self._sample_frequency[str(idx)] = 1

    def log_partial_metrics(self):
        if self.latest_beta is not None:
            log_to_tensorboard("replay_buffer/beta", self.latest_beta)

    def log_summary_metrics(self):
        log_to_tensorboard("replay_buffer/indices", self._indices_summary)
        log_to_tensorboard("replay_buffer/index_frequency", list(self._sample_frequency.values()))

        if self._weights_summary is not None:
            log_to_tensorboard("replay_buffer/weights", self._weights_summary)


# TODO: we could create an logging interface that make logging generic
def log_to_tensorboard(metric_name, data):
    if type(data) is list:
        tf.compat.v2.summary.histogram(name=metric_name, data=data)
    else:
        tf.compat.v2.summary.scalar(name=metric_name, data=data)
