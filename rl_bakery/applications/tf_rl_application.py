from rl_bakery.replay_buffer.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.specs import tensor_spec, array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common


import abc
import logging
import numpy as np
import six
import tensorflow as tf


logger = logging.getLogger(__name__)


@six.add_metaclass(abc.ABCMeta)
class TFRLApplication(object):
    """
    This is an abstract class that needs to be implemented when creating custom rl pipelines. It will contain a list
    of static methods that needs to be implemented to perform tasks that are custom the use case.
    Unlike class AbstractRLApplication where build observations and reward are required, this class needs build
    timesteps function to be implemented
    """

    @property
    @abc.abstractmethod
    def env_id_cols(self):
        """Returns a list of column names whose combination represents an unique environment"""
        pass

    @property
    @abc.abstractmethod
    def ts_id_col(self):
        """id of current timestep within its episode. It can be an integer index of the timestep. It can also
        be a timestamp representing the time that the timestep happens"""
        pass

    @property
    @abc.abstractmethod
    def obs_cols(self):
        pass

    @property
    @abc.abstractmethod
    def observation_spec(self):
        pass

    @property
    @abc.abstractmethod
    def action_spec(self):
        pass

    @property
    @abc.abstractmethod
    def training_config(self):
        pass

    @abc.abstractmethod
    def build_time_steps(self, start_dt, end_dt):
        """
        Build time steps dataframe for a time window (start_dt, end_dt)

        A timestep should contain following columns:
        <env_id_cols>: a list of ids whose combination represents an unique environment where the timestep belongs to
        timestep_id: id of current timestep within its episode. It can be an integer index of the timestep. It can also
        be a timestamp representing the time that the timestep happens
        reward: reward received at the beginning of this timestep (presumably due to actions taken in previous
        timesteps.) This value is ignored for the first timestep of an episode.

        action: action applied
        step_type: step type. It can be first, middle and last
        ob_<feature_name>: each column represents a feature

        Params:
            start_dt: timestep start datetime
            end_dt: timestep end datetime

        Returns:
            a dataframe of timesteps
        """
        pass

    def init_agent(self):
        """ a DQN agent is set by default in the application"""
        # get the global step
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # TODO: update this to get the optimizer from tensorflow 2.0 if possible
        optimizer = tf.compat.v1.train.AdamOptimizer()

        q_net = q_network.QNetwork(self.observation_spec, self.action_spec)
        time_step_spec = ts.time_step_spec(self.observation_spec)
        tf_agent = dqn_agent.DqnAgent(
            time_step_spec,
            self.action_spec,
            q_network=q_net,
            optimizer=optimizer,
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

    def init_replay_buffer(self):
        agent = self.init_agent()
        return TFUniformReplayBuffer(agent.collect_data_spec)


class MockRLApplication(TFRLApplication):
    _agent_config = {}
    test_agent = None
    _env_id_cols = None
    _ts_id_col = None
    @property
    def env_id_cols(self):
        """Returns a list of column names whose combination represents an unique environment"""
        return self._env_id_cols if self._env_id_cols else ["env_col_1"]

    @env_id_cols.setter
    def env_id_cols(self, value):
        logger.info('Setting value to %s', str(value))
        self._env_id_cols = value

    @property
    def ts_id_col(self):
        """id of current timestep within its episode. It can be an integer index of the timestep. It can also
        be a timestamp representing the time that the timestep happens"""
        return self._ts_id_col if self._ts_id_col else ["ts_col_1"]

    @ts_id_col.setter
    def ts_id_col(self, value):
        logger.info('Setting value to %s', str(value))
        self._ts_id_col = value

    @property
    def obs_cols(self):
        return ["obs_col_1"]

    @property
    def observation_spec(self):
        return tensor_spec.from_spec(array_spec.BoundedArraySpec((1,), np.int32, minimum=[1], maximum=[2]))

    @property
    def action_spec(self):
        return tensor_spec.from_spec(array_spec.BoundedArraySpec((), np.int32, minimum=0, maximum=2))

    @property
    def training_config(self):
        if not self._agent_config:
            return {
                "fc_layer_params": (100,),
                "epsilon_greediness": 0.1,
                "learning_rate": 0.01
            }
        else:
            return self._agent_config

    @training_config.setter
    def agent_config(self, value):
        logger.info('Setting value to %s', str(value))
        self._agent_config = value

    def build_time_steps(self, start_dt, end_dt):
        mock_timestep = [{"env_col_1": 1, "ts_col_1": 1, "obs_col_1": 1, "action": 1, "reward": 0.0, "step_type": 0}]
        mock_timestep_df = self.spark.createDataFrame(mock_timestep)
        return mock_timestep_df

    def init_agent(self):
        if self.test_agent:
            return self.test_agent

        self.test_agent = super().init_agent()
        return self.test_agent
