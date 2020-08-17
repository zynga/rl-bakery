from datetime import datetime, timedelta

from rl_bakery.applications.tfenv.tf_env_engine_conf import TFEnvEngineConfig
from rl_bakery.applications.tfenv.tf_env_rl_application import TFEnvRLApplication
from rl_bakery.applications.tfenv.indexed_tf_env import IndexedTFEnv
from rl_bakery.data_manager.data_manager import DATANAME
from rl_bakery.engine.base_engine import BaseEngine
from rl_bakery.operation.base_operation import start_tensorboard_writer, close_tensorboard_writer

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import q_network
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from tf_agents.agents import tf_agent
from tf_agents.drivers import driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.bandits.environments.bandit_py_environment import BanditPyEnvironment

import logging
import tensorflow as tf
import numpy as np
import time

logger = logging.getLogger(__name__)


class TwoWayPyEnvironment(BanditPyEnvironment):

    def __init__(self):
        action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=2, name='action')
        observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=-2, maximum=2, name='observation')

        # Flipping the sign with probability 1/2.
        self._reward_sign = 2 * np.random.randint(2) - 1
        print("reward sign:")
        print(self._reward_sign)

        super(TwoWayPyEnvironment, self).__init__(observation_spec, action_spec)

    def _observe(self):
        self._observation = np.random.randint(-2, 3, (1,), dtype='int32')
        return self._observation

    def _apply_action(self, action):
        return self._reward_sign * action * self._observation[0]


class TwoWaySignPolicy(tf_policy.Base):
    def __init__(self, situation, obs_spec, action_spec, ts_spec):
        observation_spec = obs_spec
        action_spec = action_spec
        time_step_spec = ts_spec
        self._situation = situation
        super(TwoWaySignPolicy, self).__init__(time_step_spec=time_step_spec,
                                            action_spec=action_spec)

    def _distribution(self, time_step):
        pass

    def _variables(self):
        return [self._situation]

    def _action(self, time_step, policy_state, seed):
        sign = tf.cast(tf.sign(time_step.observation[0, 0]), dtype=tf.int32)

        def case_unknown_fn():
            # Choose 1 so that we get information on the sign.
            return tf.constant(1, shape=(1,))

        # Choose 0 or 2, depending on the situation and the sign of the observation.
        def case_normal_fn():
            return tf.constant(sign + 1, shape=(1,))

        def case_flipped_fn():
            return tf.constant(1 - sign, shape=(1,))

        cases = [(tf.equal(self._situation, 0), case_unknown_fn),
                    (tf.equal(self._situation, 1), case_normal_fn),
                    (tf.equal(self._situation, 2), case_flipped_fn)]
        action = tf.case(cases, exclusive=True)
        return policy_step.PolicyStep(action, policy_state)


class SignAgent(tf_agent.TFAgent):
    def __init__(self, obs_spec, action_spec, ts_spec):
        self._situation = tf.compat.v2.Variable(0, dtype=tf.int32)
        policy = TwoWaySignPolicy(self._situation, obs_spec, action_spec, ts_spec)
        time_step_spec = policy.time_step_spec
        action_spec = policy.action_spec
        super(SignAgent, self).__init__(time_step_spec=time_step_spec,
                                        action_spec=action_spec,
                                        policy=policy,
                                        collect_policy=policy,
                                        train_sequence_length=None)

    def _initialize(self):
        return tf.compat.v1.variables_initializer(self.variables)

    def _train(self, experience, weights=None):
        observation = experience.observation
        action = experience.action
        reward = experience.reward

        # We only need to change the value of the situation variable if it is
        # unknown (0) right now, and we can infer the situation only if the
        # observation is not 0.
        needs_action = tf.logical_and(tf.equal(self._situation, 0),
                                    tf.not_equal(reward, 0))

        def new_situation_fn():
            """This returns either 1 or 2, depending on the signs."""
            tensor = (3 - tf.sign(tf.cast(observation[0, 0, 0], dtype=tf.int32) *
                                tf.cast(action[0, 0], dtype=tf.int32) *
                                tf.cast(reward[0, 0], dtype=tf.int32))) / 2
            tensor = tf.cast(tensor, dtype=tf.int32)
            return tensor

        new_situation = tf.cond(needs_action,
                                new_situation_fn,
                                lambda: self._situation)
        new_situation = tf.cast(new_situation, tf.int32)
        tf.compat.v1.assign(self._situation, new_situation)
        return tf_agent.LossInfo((), ())

class ExampleCartPole(object):

    def __init__(self,
                 # Params Q network
                 fc_layer_params=(100,),
                 # Params for training
                 learning_rate=0.01,
                 agent_discount=0.99,
                 mini_batch_size=1,
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
            "n_step": 1,
            "num_iterations": num_iterations,
            # "agent_discount": agent_discount,
            "mini_batch_size": mini_batch_size,
            # "eps_start": eps_start,
            # "eps_final": eps_final,
            # "eps_steps": eps_steps,
            "initial_collect_steps": initial_collect_steps,
            "log_interval": log_interval
        }
        self._rl_app = TFEnvRLApplication(envs, training_config, steps_num_per_run, start_dt, training_interval)

        @staticmethod
        def init_agent():
            """ a DQN agent is set by default in the application"""
            time_step_spec = ts.time_step_spec(self._rl_app.observation_spec)
            agent = SignAgent(self._rl_app.observation_spec, self._rl_app.action_spec, time_step_spec)
            agent.initialize()
            logger.info("tf_agent initialization is complete")

            # Optimize by wrapping some of the code in a graph using TF function.
            agent.train = common.function(agent.train)

            return agent

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

        eval_env = ExampleCartPole._make_env()

        average_reward = ExampleCartPole._compute_avg_return(eval_env, trained_policy, num_eval_episodes)
        logger.info("step = {}: eval average reward = {}".format(run_id, average_reward))

        tb_writer = start_tensorboard_writer(self._engine_config.tensorboard_path, int(run_id / self._eval_interval))
        tf.compat.v2.summary.scalar(name="eval_avg_rwd", data=average_reward)
        close_tensorboard_writer(tb_writer)

        return average_reward

    @staticmethod
    def _make_env():
        # function to create a tf environment
        # import pdb; pdb.set_trace()
        return tf_py_environment.TFPyEnvironment(TwoWayPyEnvironment())

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
    example_pipeline = ExampleCartPole(tb_path='/tmp/rl_application/mab_test_v2')
    results = example_pipeline.run()
    import pdb; pdb.set_trace()
