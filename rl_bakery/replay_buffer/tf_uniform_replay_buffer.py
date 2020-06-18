from rl_bakery.replay_buffer.tf_replay_buffer_abstract import TFReplayBufferAbstract, build_tf_trajectory
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer as TFReplayBuffer
import tensorflow as tf


class TFUniformReplayBuffer(TFReplayBufferAbstract):

    def _init_replay_buffer(self, batch_size, data_spec):
        self._batch_size = batch_size
        buffer_config = {
            "batch_size": self._batch_size,
            "data_spec": data_spec,
            "max_length": 1
        }
        tf.compat.v2.summary.scalar(name="replay_buffer_size", data=self._batch_size)
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

        if batch_size is None:
            batch_size = self._batch_size

        # TODO: convert the replay buffer to a dataset and iterate over it
        traj, _ = self._replay_buffer.get_next(sample_batch_size=batch_size)
        return traj, None
