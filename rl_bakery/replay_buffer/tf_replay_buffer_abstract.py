from rl_bakery.replay_buffer.replay_buffer_abstract import ReplayBufferAbstract
from tf_agents.trajectories import trajectory as tj
from tensorflow import TensorShape

import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class TFReplayBufferAbstract(ReplayBufferAbstract):
    _replay_buffer = None
    _collect_data_spec = None

    def __init__(self, collect_data_spec):
        self._collect_data_spec = collect_data_spec

    @property
    def collect_data_spec(self):
        return self._collect_data_spec

    @collect_data_spec.setter
    def collect_data_spec(self, value):
        self._collect_data_spec = value

    @abc.abstractmethod
    def add_batch(self, traj_dict):
        """
        add a trajectory to the replay buffer

        Params
            traj (dict[dim]:numpy): a dict of tensors representing the trajectory to be added it to the replay buffer
        """
        pass

    @abc.abstractmethod
    def get_batch(self, mini_batch_size):
        pass


def build_tf_trajectory(traj_dict, data_spec_dict):
    """
    build a trajectory of tensors based on the data and the spec provided

    Params:
        traj_dict: dict containing trajectory data stored as numpy data types
        data_spec_dict: a dict mapping every trajectory data to it's expected TensorSpec
    Return:
        tf.trajectory
        trajectory spec
    """

    traj_tensor_dict = {}
    traj_spec = {}

    for field_name, data in traj_dict.items():
        traj_tensor_dict[field_name], traj_spec[field_name] = convert_data_to_tensor(data, data_spec_dict[field_name])

    return tj.Trajectory(**traj_tensor_dict), tj.Trajectory(**traj_spec)


def convert_data_to_tensor(data, data_spec):
    """
    convert a python data object to a tensor. In case the object is a dict, it's content is converted instead.
    The data spec contains the tensorflow type to which the data will be converted. Both data data data_spec
    have the same structure

    Params:
        data:
        data_spec:
    return:
        Tensor, a dict of tensors or ()
    """
    if not data:
        return (), ()
    elif isinstance(data, dict):
        policy_dict = {}
        policy_spec = {}

        for k, data in data.items():
            policy_dict[k], policy_spec[k] = convert_data_to_tensor(data, data_spec[k])

        return policy_dict, policy_spec
    else:
        # case the tensors with their expected data type
        t_type = data_spec.dtype
        tensor_data = tf.convert_to_tensor(data, dtype=t_type)

        # extract tensor spec and remove batch dimension
        t_shape = TensorShape(tensor_data.shape.dims[1:])
        t_spec = tf.TensorSpec(t_shape, t_type)
        return tensor_data, t_spec
