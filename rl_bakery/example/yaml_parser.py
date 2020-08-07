from datetime import datetime, timedelta

from rl_bakery.applications.tfenv.tf_env_engine_conf import TFEnvEngineConfig
from rl_bakery.applications.tfenv.tf_env_rl_application import TFEnvRLApplication
from rl_bakery.applications.tfenv.indexed_tf_env import IndexedTFEnv
from rl_bakery.data_manager.data_manager import DATANAME
from rl_bakery.engine.base_engine import BaseEngine
from rl_bakery.operation.base_operation import start_tensorboard_writer, close_tensorboard_writer

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import q_network
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

import logging
import tensorflow as tf
import time
import yaml

logger = logging.getLogger(__name__)


class SimulatedEnviroment(object):

    def __init__(self, config_file_path):

        self._env_params = {}
        with open(config_file_path, 'r') as stream:
            try:
                self._env_params = yaml.safe_load(stream)
            except Exception:
                raise Exception("could not read yaml file")

        # store configs used during the training run
        self._runs_num = self._env_params['training']['runs_num']
        self._eval_interval = self._env_params['training']['eval_interval']
        self._num_eval_episodes = self._env_params['training']['num_eval_episodes']

        envs_num = self._env_params['enviroment']['envs_num']
        env_name = self._env_params['enviroment']['env_name']

        # setup rl_app
        envs = [IndexedTFEnv(self._make_env(env_name), i) for i in range(0, envs_num)]
        application_name = self._env_params['application']['name']
        version = "%s" % str(time.time())

        # setup training engine
        training_interval = timedelta(days=self._env_params['engine']['trainning_interval_days'])
        start_params = self._env_params['engine']['start_date']
        start_dt = datetime(year=start_params['year'], month=start_params['month'], day=start_params['day'], hour=start_params['hour'])
        trajectory_training_window = self._env_params['engine']['trajectory_training_window']
        self._engine_config = TFEnvEngineConfig(start_dt, training_interval, trajectory_training_window,
                                                application_name, version)
        if self._env_params['engine'].get('tb_path'):
            self._engine_config.tensorboard_path = self._env_params['engine']['tb_path']

        # setup app
        training_config = {
            "mini_batch_size": None,
        }
        training_config.update(self._env_params['training']['config'])
        steps_num_per_run = self._env_params['training']['steps_num_per_run']

        self._rl_app = TFEnvRLApplication(envs, training_config, steps_num_per_run, start_dt, training_interval)

        @staticmethod
        def init_agent():
            """ a agent is set by default in the application"""
            # get the global step
            global_step = tf.compat.v1.train.get_or_create_global_step()

            # TODO: update this to get the optimizer from tensorflow 2.0 if possible
            optimizer_kwargs = self._env_params['optimizer']['kwargs']
            optimizer_type = self._env_params['optimizer']['type']
            if optimizer_type == 'Adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(**optimizer_kwargs)
            else:  # default to Adam Optimizer
                optimizer = tf.compat.v1.train.AdamOptimizer(**optimizer_kwargs)

            agent_type = self._env_params['agent']['type']
            agent_kwargs = self._env_params['agent']['kwargs']
            time_step_spec = ts.time_step_spec(self._rl_app.observation_spec)
            if agent_type == "dqn" or agent_type == "ddqn":
                q_net_kwargs = self._env_params['agent']['q_net']['kwargs']
                q_net = q_network.QNetwork(self._rl_app.observation_spec, self._rl_app.action_spec, **q_net_kwargs)

                if agent_type == "dqn":
                    agent = dqn_agent.DqnAgent
                else:
                    agent = dqn_agent.DdqnAgent

                tf_agent = agent(
                    time_step_spec,
                    self._rl_app.action_spec,
                    q_network=q_net,
                    optimizer=optimizer,
                    gradient_clipping=None,
                    td_errors_loss_fn=common.element_wise_squared_loss,
                    train_step_counter=global_step,
                    **agent_kwargs
                )

            elif agent_type == "ddpg":
                actor_net_kwargs = self._env_params['agent']['actor_net']['kwargs']
                actor_net = actor_network.ActorNetwork(
                    self._rl_app.observation_spec,
                    self._rl_app.action_spec,
                    **actor_net_kwargs)

                critic_net_kwargs = self._env_params['agent']['critic_network']['kwargs']
                value_net = critic_network.CriticNetwork(
                    (time_step_spec.observation, self._rl_app.action_spec),
                    **critic_net_kwargs)
                tf_agent = ddpg_agent.DdpgAgent(
                    time_step_spec,
                    self._rl_app.action_spec,
                    actor_network=actor_net,
                    critic_network=value_net,
                    actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                        learning_rate=1e-4),
                    critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                        learning_rate=1e-3),
                    dqda_clipping=None,
                    td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
                    train_step_counter=global_step,
                    **agent_kwargs)

            elif agent_type == "ppo":
                actor_net = actor_distribution_network.ActorDistributionNetwork(self._rl_app.observation_spec, self._rl_app.action_spec)
                value_net = value_network.ValueNetwork(self._rl_app.observation_spec)
                tf_agent = ppo_agent.PPOAgent(
                    time_step_spec,
                    self._rl_app.action_spec,
                    optimizer,
                    actor_net=actor_net,
                    value_net=value_net,
                    train_step_counter=global_step,
                    **agent_kwargs)

            tf_agent.initialize()
            logger.info("tf_agent initialization is complete")

            # Optimize by wrapping some of the code in a graph using TF function.
            tf_agent.train = common.function(tf_agent.train)

            return tf_agent
        self._rl_app.init_agent = init_agent.__get__(object)

    def run(self):
        engine = BaseEngine(self._rl_app, self._engine_config)
        self._rl_app.set_dm(engine._dm)

        engine.init(force_run=True)

        logger.info("Training started")
        eval_avg_rwd = []
        for run_id in range(1, self._runs_num):
            engine.train(run_id)

            if run_id % self._eval_interval == 0:
                avg_rwd = self._evaluate_agent(engine._dm, run_id, self._num_eval_episodes)
                eval_avg_rwd.append(avg_rwd)

        logger.info("Training is done")
        logger.info("Eval result: %s" % str(eval_avg_rwd))
        return eval_avg_rwd

    def _evaluate_agent(self, dm, run_id, num_eval_episodes):
        rl_agent = dm.get(DATANAME.MODEL, run_id)

        trained_policy = rl_agent.policy

        eval_env = SimulatedEnviroment._make_env(self._env_params['enviroment']['env_name'])

        average_reward = SimulatedEnviroment._compute_avg_return(eval_env, trained_policy, num_eval_episodes)
        logger.info("step = {}: eval average reward = {}".format(run_id, average_reward))

        tb_writer = start_tensorboard_writer(self._engine_config.tensorboard_path, int(run_id / self._eval_interval))
        tf.compat.v2.summary.scalar(name="eval_avg_rwd", data=average_reward)
        close_tensorboard_writer(tb_writer)

        return average_reward

    @staticmethod
    def _make_env(env_name):
        # function to create a tf environment
        return tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

    @staticmethod
    def _compute_avg_return(environment, policy, num_episodes=100):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]


if __name__ == "__main__":
    example_pipeline = SimulatedEnviroment('mountaincar_ddpg.yml')
    example_pipeline.run()
