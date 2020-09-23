import abc
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents import ddpg
from tf_agents.trajectories import time_step as ts
import logging

from rl_bakery.agent_trainer.config import Optimizer

logger = logging.getLogger(__name__)


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


class QAgent(TFAgentTrainer):

    def __init__(self, data_spec, config):
        super().__init__(data_spec, config)

    def _build_q_network(self):
        # By default, this builds a q-network using the supplied fc_layer_params
        # This function can be overridden to provide a more complicated Q Network
        assert(self._config.agent.fc_layer_params)
        q_net = q_network.QNetwork(self._data_spec.observation_spec, self._data_spec.action_spec,
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


class DQNAgent(QAgent):

    def __init__(self, data_spec, config):
        super().__init__(data_spec, config)

    def _init_qagent(self, optimizer, q_net, global_step):
        time_step_spec = ts.time_step_spec(self._data_spec.observation_spec)
        return dqn_agent.DqnAgent(
            time_step_spec,
            self._data_spec.action_spec,
            q_network=q_net,
            optimizer=optimizer,
            epsilon_greedy=self._config.policy.epsilon_greedy,
            gradient_clipping=self._config.agent.gradient_clipping,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=global_step,
            debug_summaries=self._config.agent.debug_summaries,
            summarize_grads_and_vars=self._config.agent.summarize_grads_and_vars
        )


class DDQNAgent(QAgent):

    def __init__(self, data_spec, config):
        super().__init__(data_spec, config)

    def _init_qagent(self, optimizer, q_net, global_step):
        time_step_spec = ts.time_step_spec(self._data_spec.observation_spec)
        return dqn_agent.DdqnAgent(
            time_step_spec,
            self._data_spec.action_spec,
            q_network=q_net,
            optimizer=optimizer,
            epsilon_greedy=self._config.policy.epsilon_greedy,
            gradient_clipping=self._config.agent.gradient_clipping,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=global_step,
            debug_summaries=True,
            summarize_grads_and_vars=True
        )


class DDPGAgent(TFAgentTrainer):

    def __init__(self, data_spec, config):
        super().__init__(data_spec, config)

    def _build_actor_network(self):
        assert(self._config.agent.actor_fc_layer_params)
        return ddpg.actor_network.ActorNetwork(
            self._data_spec.observation_spec,
            self._data_spec.action_spec,
            fc_layer_params=self._config.agent.actor_fc_layer_params)

    def _build_critic_network(self):
        return ddpg.critic_network.CriticNetwork(
            (self._data_spec.observation_spec, self._data_spec.action_spec),
            observation_fc_layer_params=self._config.agent.observation_fc_layer_params,
            action_fc_layer_params=self._config.agent.action_fc_layer_params,
            joint_fc_layer_params=self._config.agent.joint_fc_layer_params)

    def _get_td_loss_fn(self):
        return tf.compat.v1.losses.huber_loss

    def _init_tfagent(self, global_step):
        time_step_spec = ts.time_step_spec(self._data_spec.observation_spec)
        actor_net = self._build_actor_network()
        value_net = self._build_critic_network()
        tf_agent = ddpg.ddpg_agent.DdpgAgent(
            time_step_spec,
            self._data_spec.action_spec,
            actor_network=actor_net,
            critic_network=value_net,
            actor_optimizer=self._get_optimizer(self._config.agent.actor_optimizer),
            critic_optimizer=self._get_optimizer(self._config.agent.critic_optimizer),
            ou_stddev=self._config.agent.ou_stddev,
            ou_damping=self._config.agent.ou_damping,
            target_update_tau=self._config.agent.target_update_tau,
            target_update_period=self._config.agent.target_update_period,
            dqda_clipping=self._config.agent.dqda_clipping,
            td_errors_loss_fn=self._get_td_loss_fn(),
            gamma=self._config.trajectory.agent_discount,
            reward_scale_factor=self._config.agent.reward_scale_factor,
            gradient_clipping=self._config.agent.gradient_clipping,
            debug_summaries=self._config.agent.debug_summaries,
            summarize_grads_and_vars=self._config.agent.summarize_grads_and_vars,
            train_step_counter=global_step)
        tf_agent.initialize()

        return tf_agent