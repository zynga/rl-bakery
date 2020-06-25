""" Cartpole-V0 with a ddqn agent """
from datetime import datetime, timedelta

import logging
import time
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from rl_bakery.applications.tfenv.tf_env_engine_conf import TFEnvEngineConfig
from rl_bakery.applications.tfenv.tf_env_rl_application import TFEnvRLApplication
from rl_bakery.applications.tfenv.indexed_tf_env import IndexedTFEnv
from rl_bakery.example.cartpole import ExampleCartPole

logger = logging.getLogger(__name__)


class ExampleCartPoleDDQN(ExampleCartPole):
    """ Cartpole DDQN Class """

    def __init__(self,
                 # Params Q network
                 fc_layer_params=(100,),
                 # Params for training
                 learning_rate=0.01,
                 agent_discount=0.99,
                 mini_batch_size=64,
                 num_iterations=10000,
                 gradient_clipping=None,
                 trajectory_training_window=100,
                 log_interval=200,
                 # Param for simulated environments
                 envs_num=10,
                 runs_num=10,
                 steps_num_per_run=1000,
                 # Params for evaluation
                 eval_interval=1,
                 num_eval_episodes=100,
                 # Params for data collection
                 eps_start=1.0,
                 eps_final=0.1,
                 eps_steps=10000,
                 initial_collect_steps=3000,
                 tb_path=None):

        # store configs used during the training run
        self._runs_num = runs_num
        self._eval_interval = eval_interval
        self._num_eval_episodes = num_eval_episodes

        # setup rl_app
        envs = [IndexedTFEnv(self._make_env(), i) for i in range(0, envs_num)]
        application_name = "CartPole-example"
        version = "%s" % str(time.time())

        # setup training engine
        training_interval = timedelta(days=1)
        start_dt = datetime(year=2019, month=8, day=7, hour=10)
        self._engine_config = TFEnvEngineConfig(start_dt, training_interval, trajectory_training_window,
                                                application_name, version)
        if tb_path:
            self._engine_config.tensorboard_path = tb_path
        # setup app
        training_config = {
            "num_iterations": num_iterations,
            "agent_discount": agent_discount,
            "mini_batch_size": mini_batch_size,
            "eps_start": eps_start,
            "eps_final": eps_final,
            "eps_steps": eps_steps,
            "initial_collect_steps": initial_collect_steps,
            "log_interval": log_interval
        }
        self._rl_app = TFEnvRLApplication(envs, training_config, steps_num_per_run, start_dt, training_interval)

        @staticmethod
        def init_agent():
            """ a DQN agent is set by default in the application"""
            # get the global step
            global_step = tf.compat.v1.train.get_or_create_global_step()

            # TODO: update this to get the optimizer from tensorflow 2.0 if possible
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

            q_net = q_network.QNetwork(self._rl_app.observation_spec, self._rl_app.action_spec,
                                       fc_layer_params=fc_layer_params)
            time_step_spec = ts.time_step_spec(self._rl_app.observation_spec)
            tf_agent = dqn_agent.DdqnAgent(
                time_step_spec,
                self._rl_app.action_spec,
                q_network=q_net,
                optimizer=optimizer,
                epsilon_greedy=eps_final,
                gradient_clipping=gradient_clipping,
                td_errors_loss_fn=common.element_wise_squared_loss,
                train_step_counter=global_step,
                debug_summaries=True,
                summarize_grads_and_vars=True
            )
            tf_agent.initialize()
            logger.info("tf_agent initialization is complete")

            # Optimize by wrapping some of the code in a graph using TF function.
            tf_agent.train = common.function(tf_agent.train)

            return tf_agent

        self._rl_app.init_agent = init_agent.__get__(object)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_pipeline = ExampleCartPoleDDQN()
    example_pipeline.run()
