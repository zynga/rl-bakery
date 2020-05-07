import abc
import logging
import six

logger = logging.getLogger(__name__)


@six.add_metaclass(abc.ABCMeta)
class AbstractRLApplication(object):
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

    @abc.abstractmethod
    def init_agent(self):
        pass

    @abc.abstractmethod
    def init_replay_buffer(self):
        pass
