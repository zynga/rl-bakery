from omegaconf import MISSING
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


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


@dataclass
class QConfig(AgentConfig):
    optimizer: OptimizerConfig = OptimizerConfig()
    summarize_grads_and_vars: bool = False
    debug_summaries: bool = False
    gradient_clipping: Optional[float] = None
    fc_layer_params: List[int] = MISSING


@dataclass()
class DDPGConfig(AgentConfig):
    actor_fc_layer_params: List[int] = MISSING
    observation_fc_layer_params: List[int] = MISSING
    action_fc_layer_params: Optional[List[int]] = None
    joint_fc_layer_params: List[int] = MISSING
    actor_optimizer: OptimizerConfig = OptimizerConfig()
    critic_optimizer: OptimizerConfig = OptimizerConfig()
    ou_stddev: float = 0.2
    ou_damping: float = 0.15
    target_update_tau: float = 0.05
    target_update_period: int = 5
    dqda_clipping: Optional[float] = None
    reward_scale_factor: float = 1.0
    gradient_clipping: Optional[float] = None
    debug_summaries: bool = False
    summarize_grads_and_vars: bool = False
    #TODO add td_errors_loss_fn
