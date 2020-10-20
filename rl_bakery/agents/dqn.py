from rl_bakery.agents.abstract import QAgent, AgentConfig, OptimizerConfig
from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.trajectories import time_step as ts
import logging
from omegaconf import MISSING
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class QConfig(AgentConfig):
    optimizer: OptimizerConfig = OptimizerConfig()
    fc_layer_params: List[int] = MISSING # If this isn't provided, then _build_q_network() must be overwritten

    boltzmann_temperature: Optional[int] = MISSING
    emit_log_probability: Optional[bool] = MISSING
    # Params for target network updates
    target_update_tau: Optional[float] = MISSING
    target_update_period: Optional[int] = MISSING
    # Params for training.
    gamma: Optional[float] = MISSING
    reward_scale_factor: Optional[float] = MISSING
    gradient_clipping: Optional[float] = MISSING
    # Params for debugging
    debug_summaries: Optional[bool] = MISSING
    summarize_grads_and_vars: Optional[bool] = MISSING
    name: Optional[str] = MISSING


class DQNAgent(QAgent):

    def __init__(self, data_spec, config):
        super().__init__(data_spec, config)

    def _observation_and_action_constraint_splitter(self):
        return None

    def _td_errors_loss_fn(self):
        return None

    def _args_to_ignore(self):
        return ['fc_layer_params', 'optimizer', 'fc_layer_params']

    def _get_agent_args(self, optimizer, q_net, global_step):
        time_step_spec = ts.time_step_spec(self._data_spec.observation_spec)
        args = {}
        args['q_network'] = q_net
        args['train_step_counter'] = global_step
        args['optimizer'] = optimizer
        args['time_step_spec'] = time_step_spec
        args['action_spec'] = self._data_spec.action_spec
        args['epsilon_greedy'] = self._config.policy.epsilon_greedy
        args['n_step_update'] = self._config.trajectory.n_step
        self._add_agent_args(args)
        return args

    def _init_qagent(self, optimizer, q_net, global_step):
        args = self._get_agent_args(optimizer, q_net, global_step)
        return DqnAgent(**args)


class DDQNAgent(DQNAgent):

    def __init__(self, data_spec, config):
        super().__init__(data_spec, config)

    def _init_qagent(self, optimizer, q_net, global_step):
        args = self._get_agent_args(optimizer, q_net, global_step)
        return DdqnAgent(**args)

