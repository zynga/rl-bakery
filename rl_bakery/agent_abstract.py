import abc
import six


@six.add_metaclass(abc.ABCMeta)
class AgentAbstract(object):
    """Abstract base class for RL agents."""

    @abc.abstractmethod
    def train(self, experience, weights):
        """
        Train the agent

        Params:
            experience: a batch of trajectories to be use to train the agent with.
            weights: a list/tensor of weights to be used when computing the traing loss of
                the trajectories in the "experience"

        Return:
             LossInfo: an object containing information of the loss experienced during the training
        """
        pass

    @property
    @abc.abstractmethod
    def policy(self):
        """
        Return the agent's current optimal policy.
        """
        pass

    @property
    @abc.abstractmethod
    def collect_policy(self):
        """
        Return the agent's policy used for exploration.
        """
        pass

    @property
    @abc.abstractmethod
    def collect_data_spec(self):
        """
        Return the spec of the data that the collect policy expects.
        """
        pass


class MockAgent(AgentAbstract):
    def train(self, experience, weights):
        pass

    def policy(self):
        return "policy"

    def collect_policy(self):
        return "collect_policy"

    def collect_data_spec(self):
        return "collect_data_spec"
