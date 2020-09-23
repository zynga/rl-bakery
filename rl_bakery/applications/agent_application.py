from omegaconf import OmegaConf, MISSING, SI
from rl_bakery.engine.timing_data import TimingData
from dataclasses import dataclass
from rl_bakery.replay_buffer.tf_uniform_replay_buffer import TFUniformReplayBuffer
import math
from typing import Optional
from dataclasses import dataclass
from rl_bakery.agent_trainer.config import AgentConfig
from rl_bakery.agent_trainer.agents import AgentTrainer

from datetime import timedelta, datetime


@dataclass
class PolicyConfig:
    pass


@dataclass
class EpsilonPolicyConfig(PolicyConfig):
    epsilon_greedy: float = 0.1
    # Parameters to tune epsilon over epochs
    eps_start: float = SI("${policy.epsilon_greedy}")
    eps_final: float = SI("${policy.epsilon_greedy}")
    eps_steps: int = 10000
    initial_collect_steps: int = 3000


@dataclass
class ProjectConfig:
    application_name: str = MISSING
    tensorboard_path: str = MISSING
    dm_storage_path: str = MISSING
    version: str = "v1"
    log_interval: int = 200
    summary_interval: int = SI("${project.log_interval}")


@dataclass
class TrajectoryConfig:
    n_step: int = 1
    agent_discount: float = 0.99
    trajectory_training_window: int = 1 # how many previous timesteps to include in training traj


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_iterations: int = 1000


@dataclass
class EnvConfig:
    num_steps_per_run: Optional[int] = 1000
    num_envs: Optional[int] = 10


@dataclass
class ApplicationConfig:
    training: TrainingConfig
    agent: AgentConfig
    project: ProjectConfig
    policy: PolicyConfig
    trajectory: TrajectoryConfig
    env: EnvConfig


@dataclass
class DataSpec:
    observation_spec: 'typing.Any'
    action_spec: 'typing.Any'


@dataclass
class AgentApplication:
    data_spec: DataSpec
    agent_trainer: AgentTrainer
    config: ApplicationConfig
    env: 'typing.Any'

    # TODO: consider make this a TimeData class
    first_timestep_dt: datetime
    training_interval: timedelta
    observation_offset: timedelta = timedelta(days=0)

    @property
    def training_timestep_lag(self):
        """The delay of how long before a timestep can be used for training (in timesteps)"""
        return int(math.ceil((self.observation_offset / self.training_interval)))

    @property
    def timing_data(self):
        return TimingData(start_dt=self.first_timestep_dt,
                          training_interval=self.training_interval,
                          training_timestep_lag=self.training_timestep_lag,
                          trajectory_training_window=self.config.trajectory.trajectory_training_window)

    # TODO switch this to an attribute of AgentApplication
    def init_replay_buffer(self):
        # TODO: see if there's a way to pass in an initialized agent
        agent = self.agent_trainer.init_agent()
        return TFUniformReplayBuffer(agent.collect_data_spec)


def make_config(agent_config, dotlist):
    """
    This creates a configuration for an AgentApplication. It adds a given dotlist of arguments to an ApplicationConfig.

    :param agent_config: an AgentConfig for the application's Agent
    :param dotlist: A list of parameter values, eg. ["training.learning_rate=0.01", "training.num_iterations=10000"]
    :return: An OmegaConf representing the configuration
    """
    conf_schema = ApplicationConfig(
        training=TrainingConfig(),
        agent=agent_config,
        project=ProjectConfig(),
        trajectory=TrajectoryConfig(),
        policy=EpsilonPolicyConfig(),
        env=EnvConfig()
    )
    conf = OmegaConf.structured(conf_schema)

    conf.merge_with_dotlist(dotlist)
    return conf

