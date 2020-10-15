import abc
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.networks.q_network import QNetwork
from enum import Enum
from dataclasses import dataclass
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)


class Optimizer(Enum):
    Adam = "Adam"
    RMSprop = "RMSprop"
    Adagrad = "Adagrad"


@dataclass
class OptimizerConfig:
    optimizer: Optimizer = Optimizer.Adam
    learning_rate: float = 0.01


@dataclass
class AgentConfig:
    pass


class AgentTrainer(metaclass=abc.ABCMeta):
    """
    This set ups an Agent to be trained.
    """

    def __init__(self, data_spec, config):
        self._data_spec = data_spec
        self._config = config

    @abc.abstractmethod
    def init_agent(self):
        pass


class TFAgentTrainer(AgentTrainer):

    def __init__(self, data_spec, config):
        super().__init__(data_spec, config)

    @abc.abstractmethod
    def _init_tfagent(self, optimizer):
        pass

    def _get_optimizer(self, optimizer_config):

        # TODO use tf2 optimizers, support other optimizers
        if optimizer_config.optimizer == Optimizer.Adam:
            return tf.compat.v1.train.AdamOptimizer(learning_rate=optimizer_config.learning_rate)
        elif optimizer_config.optimizer == Optimizer.RMSprop:
            return tf.compat.v1.train.RMSPropOptimizer(learning_rate=optimizer_config.learning_rate)
        else:
            raise Exception("Unknown optimizer %s" % optimizer_config.optimizer)

    def init_agent(self):
        # get the global step
        global_step = tf.compat.v1.train.get_or_create_global_step()
        return self._init_tfagent(global_step)

    def _args_to_ignore(self):
        return []

    def _add_agent_args(self, args):
        for arg in self._config['agent'].keys():
            if arg not in self._args_to_ignore() and not OmegaConf.is_missing(self._config.agent, arg):
                args[arg] = self._config['agent'][arg]


class QAgent(TFAgentTrainer):

    def __init__(self, data_spec, config):
        super().__init__(data_spec, config)

    def _build_q_network(self):
        # By default, this builds a q-network using the supplied fc_layer_params
        # This function can be overridden to provide a more complicated Q Network
        assert(self._config.agent.fc_layer_params)
        q_net = QNetwork(self._data_spec.observation_spec, self._data_spec.action_spec,
                        fc_layer_params=self._config.agent.fc_layer_params)
        return q_net

    def _init_tfagent(self, global_step):
        q_net = self._build_q_network()
        optimizer = self._get_optimizer(self._config.agent.optimizer)
        tf_agent = self._init_qagent(optimizer, q_net, global_step)
        tf_agent.initialize()
        logger.info("tf_agent initialization is complete")

        # Optimize by wrapping some of the code in a graph using TF function.
        tf_agent.train = common.function(tf_agent.train)

        return tf_agent

    @abc.abstractmethod
    def _init_qagent(self, optimizer, qnet, global_step):
        pass
