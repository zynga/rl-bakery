""" MountainCarContinuous-V0 with a ddpg agent """
from datetime import datetime, timedelta

from rl_bakery.applications.tfenv.tf_env_engine_conf import TFEnvEngineConfig
from rl_bakery.applications.tfenv.tf_env_rl_application import TFEnvRLApplication
from rl_bakery.applications.tfenv.indexed_tf_env import IndexedTFEnv
from rl_bakery.data_manager.data_manager import DATANAME
from rl_bakery.engine.base_engine import BaseEngine
from rl_bakery.operation.base_operation import start_tensorboard_writer, close_tensorboard_writer

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.trajectories import time_step
from tf_agents.utils import common

import logging
import tensorflow as tf
import time

logger = logging.getLogger(__name__)


class ExampleMountainCarDDPG(object):

    def __init__(self,
                 # Params for training
                 learning_rate=0.001,
                 discount=0.99,
                 mini_batch_size=64,
                 num_iterations=2000,
                 gradient_clipping=None,
                 trajectory_training_window=1,
                 log_interval=100,
                 # Param for simulated environments
                 envs_num=60,
                 runs_num=10,
                 steps_num_per_run=50,
                 # Params for evaluation
                 eval_interval=1,
                 num_eval_episodes=100,
                 # Params for data collection
                 eps_start=1.0,
                 eps_final=0.1,
                 eps_steps=1000,
                 initial_collect_steps=1000,
                 tb_path=None):

        # store configs used during the training run
        self._runs_num = runs_num
        self._eval_interval = eval_interval
        self._num_eval_episodes = num_eval_episodes

        # setup rl_app
        envs = [IndexedTFEnv(self._make_env(), i) for i in range(0, envs_num)]
        application_name = "MountainCar-example"
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
            """ a DDPG agent is set by default in the application"""
            # get the global step
            global_step = tf.compat.v1.train.get_or_create_global_step()

            # TODO: update this to get the optimizer from tensorflow 2.0 if possible
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            time_step_spec = time_step.time_step_spec(self._rl_app.observation_spec)
            actor_net = actor_network.ActorNetwork(
                self._rl_app.observation_spec,
                self._rl_app.action_spec,
                fc_layer_params=(400, 300))
            value_net = critic_network.CriticNetwork(
                (time_step_spec.observation, self._rl_app.action_spec),
                observation_fc_layer_params=(400,),
                action_fc_layer_params=None,
                joint_fc_layer_params=(300,))
            tf_agent = ddpg_agent.DdpgAgent(
                time_step_spec,
                self._rl_app.action_spec,
                actor_network=actor_net,
                critic_network=value_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=1e-4),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=1e-3),
                ou_stddev=0.2,
                ou_damping=0.15,
                target_update_tau=0.05,
                target_update_period=5,
                dqda_clipping=None,
                td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
                gamma=discount,
                reward_scale_factor=1.0,
                gradient_clipping=gradient_clipping,
                debug_summaries=True,
                summarize_grads_and_vars=True,
                train_step_counter=global_step)
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

        eval_env = self._make_env()

        average_reward = self._compute_avg_return(eval_env, trained_policy, num_eval_episodes)
        logger.info("step = {}: eval average reward = {}".format(run_id, average_reward))

        tb_writer = start_tensorboard_writer(self._engine_config.tensorboard_path, int(run_id / self._eval_interval))
        def lazy():
            tf.compat.v2.summary.scalar(name="eval_avg_rwd", data=average_reward)
        lazy()
        close_tensorboard_writer(tb_writer)

        return average_reward

    @staticmethod
    def _make_env():
        # function to create a tf environment
        return tf_py_environment.TFPyEnvironment(suite_gym.load("MountainCarContinuous-v0"))

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
    logging.basicConfig(level=logging.INFO)
    example_pipeline = ExampleMountainCarDDPG()
    example_pipeline.run()
