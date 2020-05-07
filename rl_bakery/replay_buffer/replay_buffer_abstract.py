import abc
import six


@six.add_metaclass(abc.ABCMeta)
class ReplayBufferAbstract(object):
    """
    Abstract class for replay buffer
    """
    _replay_buffer = None

    @abc.abstractmethod
    def add_batch(self, traj_dict):
        """
        add a trajectory to the replay buffer. traj_dict contains a mapping from trajectory fields to their
        respective data.

        List of fields in a trajectory: [step_type, observation, action, policy_info, reward, discount, next_step_type]
        The field data will formatted in this shape: [B * T * S] where B is the batch size, T is the number of timestep
        (n-step) and S is the shape of the data point specific to that field.

        Params
            traj_dict (dict): a dict representing the trajectory to be added it to the replay buffer
        """
        pass

    @abc.abstractmethod
    def get_batch(self, batch_size):
        """
        Return a batch of items from the replay buffer of size "batch_size"

        Params:
            batch_size (int): number of items to return
        """
        pass

    def pre_process(self, curr_iter):
        """
        Prepare the replay buffer before being request to returned a batch. This is called prior to
        every "get_batch" requests.

        Params:
            curr_iter (int): current training iteration number
        """
        pass

    def post_process(self, traj_meta, loss_info, curr_iter):
        """
        Update the replay buffer state based on the new information received after training the agent.
        This is called subsequent to every train operation.

        Params:
            traj_meta: metadata for the batch used for training. This is produced by the get_batch method
            loss_info: information about the training loss produced by the agent during training
            curr_iter: current training iteration number
        """
        pass
