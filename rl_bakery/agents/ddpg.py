from rl_bakery.agents.abstract import TFAgent, AgentConfig, OptimizerConfig
from tf_agents.agents.ddpg.ddpg_agent import DdpgAgent
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.trajectories import time_step as ts
from omegaconf import MISSING
from typing import Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass()
class DDPGConfig(AgentConfig):
    actor_fc_layer_params: Optional[List[int]] = MISSING
    observation_fc_layer_params: Optional[List[int]] = None
    action_fc_layer_params: Optional[List[int]] = None
    joint_fc_layer_params: Optional[List[int]] = None
    actor_optimizer: OptimizerConfig = OptimizerConfig()
    critic_optimizer: OptimizerConfig = OptimizerConfig()
    ou_stddev: Optional[float] = MISSING
    ou_damping: Optional[float] = MISSING
    target_update_tau: Optional[float] = MISSING
    target_update_period: Optional[int] = MISSING
    dqda_clipping: Optional[float] = MISSING
    reward_scale_factor: Optional[float] = MISSING
    gradient_clipping: Optional[float] = MISSING
    debug_summaries: Optional[bool] = MISSING
    summarize_grads_and_vars: Optional[bool] = MISSING
    name: Optional[str] = MISSING


class DDPGAgent(TFAgent):

    def __init__(self, data_spec, config):
        super().__init__(data_spec, config)

    def _build_actor_network(self):
        assert(self._config.agent.actor_fc_layer_params)
        return ActorNetwork(
            self._data_spec.observation_spec,
            self._data_spec.action_spec,
            fc_layer_params=self._config.agent.actor_fc_layer_params)

    def _build_critic_network(self):
        assert(self._config.agent.observation_fc_layer_params)
        assert(self._config.agent.joint_fc_layer_params)
        return CriticNetwork(
            (self._data_spec.observation_spec, self._data_spec.action_spec),
            observation_fc_layer_params=self._config.agent.observation_fc_layer_params,
            action_fc_layer_params=self._config.agent.action_fc_layer_params,
            joint_fc_layer_params=self._config.agent.joint_fc_layer_params)

    def _build_target_actor_network(self):
        # Override this to not use TFAgent default
        return None

    def _build_target_critic_network(self):
        # Override this to not use TFAgent default
        return None

    def _get_td_loss_fn(self):
        # Override this to not use TFAgent default
        return None

    def _args_to_ignore(self):
        return ["actor_fc_layer_params", "observation_fc_layer_params", "action_fc_layer_params",
                "joint_fc_layer_params", "actor_optimizer", "critic_optimizer"]

    def _get_agent_args(self, global_step):

        time_step_spec = ts.time_step_spec(self._data_spec.observation_spec)
        args = {}
        args['time_step_spec'] = time_step_spec
        args['actor_optimizer'] = self._get_optimizer(self._config.agent.actor_optimizer)
        args['critic_optimizer'] = self._get_optimizer(self._config.agent.critic_optimizer)
        args['action_spec'] = self._data_spec.action_spec
        args['actor_network'] = self._build_actor_network()
        args['critic_network'] = self._build_critic_network()
        args['target_actor_network'] = self._build_target_actor_network()
        args['target_critic_network'] = self._build_target_critic_network()
        args['train_step_counter'] = global_step
        args['td_errors_loss_fn'] = self._get_td_loss_fn()
        self._add_agent_args(args)
        return args

    def _init_tfagent(self, global_step):
        args = self._get_agent_args(global_step)
        return DdpgAgent(**args)

