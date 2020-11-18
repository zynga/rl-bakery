import logging
import tensorflow as tf

from rl_bakery.data_manager.data_manager import DATANAME
from rl_bakery.applications.tfenv.tf_env_rl_application import TFEnvRLApplication
from rl_bakery.applications.tfenv.indexed_tf_env import IndexedTFEnv
from rl_bakery.engine.base_engine import BaseEngine
from rl_bakery.operation.base_operation import start_tensorboard_writer, close_tensorboard_writer

logger = logging.getLogger(__name__)


class SimulationRunner(object):
    """
    This trains an Agent using a simulated environment.
    """

    def __init__(self, application, make_eval_env, dm, num_runs,
                 num_eval_episodes=100,
                 eval_interval=1):
        """

        :param application: The AgentApplication
        :param make_eval_env: A function that returns an environment instance with a step(action) function which returns
        a TimeStep
        :param dm: The DataManager
        :param num_runs: The number of training cycles to go through
        :param num_eval_episodes: The number of episodes to evaluate a Agent on
        :param eval_interval: The interval between training runs and evaluations (1 means every training does an eval)
        """

        self._num_eval_episodes = num_eval_episodes
        self._num_runs = num_runs
        self._eval_interval = eval_interval
        self._application = application
        self._dm = dm
        self._make_env = make_eval_env

    def run(self):
        engine = BaseEngine(self._application, self._dm)

        engine.init(force_run=True)

        logger.info("Training started")
        eval_avg_rwd = []
        for run_id in range(1, self._num_runs):
            engine.train(run_id)

            if run_id % self._eval_interval == 0:
                avg_rwd = self._evaluate_agent(run_id, self._num_eval_episodes)
                eval_avg_rwd.append(avg_rwd)

        logger.info("Training is done")
        logger.info("Eval result: %s" % str(eval_avg_rwd))
        return eval_avg_rwd

    def _evaluate_agent(self, run_id, num_eval_episodes):
        rl_agent = self._dm.get(DATANAME.MODEL, run_id)

        trained_policy = rl_agent.policy

        eval_env = self._make_env()

        average_reward = self._compute_avg_return(eval_env, trained_policy, num_eval_episodes)
        logger.info("step = {}: eval average reward = {}".format(run_id, average_reward))

        tb_writer = start_tensorboard_writer(self._application.config.project.tensorboard_path,
                                             int(run_id / self._eval_interval))
        tf.compat.v2.summary.scalar(name="eval_avg_rwd", data=average_reward)
        close_tensorboard_writer(tb_writer)

        return average_reward

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


def make_batch_tfenv(make_env, config, start_dt, training_interval, spark_session):
    """
    This returns a TFEnvRLApplication, which keeps 'config.env.num_envs' envs running in parallel.

    :param make_env: A function that returns an Environment
    :param config: An ApplicationConfig
    :param start_dt: A datetime being used to simulate the first action
    :param training_interval: A timedelta indicating the lag between when an observation is generated and when it can
                           be used for training. This simulates real world environments where there's a delay between
                           data collection and Agent updates.
    :param spark_session: A Spark session
    :return:
    """

    envs = [IndexedTFEnv(make_env(), i) for i in range(0, config.env.num_envs)]

    # setup app
    training_config = {
        "num_iterations": config.training.num_iterations,
        "agent_discount": config.trajectory.agent_discount,
        "mini_batch_size": config.training.batch_size,
        "eps_start": config.policy.eps_start,
        "eps_final": config.policy.eps_final,
        "eps_steps": config.policy.eps_steps,
        "initial_collect_steps": config.policy.initial_collect_steps,
        "log_interval": config.project.log_interval
    }
    return TFEnvRLApplication(envs, spark_session, training_config, config.env.num_steps_per_run, start_dt, training_interval)

