from datetime import datetime, timedelta
from rl_bakery.data_manager.data_manager import DataManager, DATANAME, InMemoryStorage, TFAgentStorage

import abc
import math
import six
import logging

logger = logging.getLogger(__name__)


@six.add_metaclass(abc.ABCMeta)
class AbstractEngineConfig(object):
    """
    The EngineConfig contains configuration on how to execute an RLApplication over its lifetime.

    TODO: add more details
    """
    _dm_base_path = "/tmp/rl_application"
    _tensorboard_path = _dm_base_path

    @property
    @abc.abstractmethod
    def application_name(self):
        """Return name of the application"""
        pass

    @property
    @abc.abstractmethod
    def version(self):
        """Return version of the application (i.e v1)"""
        pass

    @property
    @abc.abstractmethod
    def start_dt(self):
        """Return a datetime representing the start date of the pipeline"""
        pass

    @property
    @abc.abstractmethod
    def training_interval(self):
        pass

    @property
    @abc.abstractmethod
    def trajectory_training_window(self):
        pass

    @property
    def observation_offset(self):
        """A timedelta representing the time between when a timestep occurs and its latest observation time."""
        return timedelta(days=0)

    @property
    def training_timestep_lag(self):
        """The delay of how long before a timestep can be used for training (in timesteps)"""
        return int(math.ceil((self.observation_offset / self.training_interval)))

    @property
    def tensorboard_path(self):
        """Tensorboard will use this path to log metrics to"""
        return "{}/{}/{}/tensorboard/".format(self._tensorboard_path, self.application_name, self.version)

    @tensorboard_path.setter
    def tensorboard_path(self, value):
        logger.info('Setting value to %s', str(value))
        self._tensorboard_path = value

    def build_data_manager(self, rl_app):
        """
        Setup the data manager which will be used to read and write data to

        Return: DataManager instance
        """
        run_time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.now())
        storage_path = "{}/{}/{}/{}/".format(self._dm_base_path, self.application_name, self.version, run_time)

        dm = DataManager()
        dm.add_data(DATANAME.MODEL, TFAgentStorage(rl_app, storage_path, "agent.model"))
        dm.add_data(DATANAME.RUN_CONTEXT, InMemoryStorage())
        dm.add_data(DATANAME.TIMESTEP, InMemoryStorage())
        return dm

    def get_current_run_id(self):
        """
        compute the run id of engine at the runtime of this function. By default, this value will be
        computed based on the application start time, current time and training_interval.

        Return:
             Int
        """
        start_datetime = self.start_dt
        run_datetime = self._get_current_datetime()
        run_interval = self.training_interval

        time_since_start = run_datetime - start_datetime
        logger.info("Time between start and run_date: %s" % str(time_since_start))

        run_id = int(time_since_start / run_interval)
        logger.info("Current run_id: %s" % str(run_id))

        return run_id

    def _get_current_datetime(self):
        return datetime.now().replace(second=0, microsecond=0)


class MockEngineConfig(AbstractEngineConfig):
    _training_timestep_lag = 0
    _application_name = ""

    @property
    def application_name(self):
        return self._application_name

    @application_name.setter
    def application_name(self, value):
        logger.info('Setting value to %s', str(value))
        self._application_name = value

    @property
    def version(self):
        return "v1"

    @property
    def start_dt(self):
        return self._start_dt

    @start_dt.setter
    def start_dt(self, value):
        logger.info('Setting value to %s', str(value))
        self._start_dt = value

    @property
    def training_interval(self):
        return self._training_interval

    @training_interval.setter
    def training_interval(self, value):
        logger.info('Setting value to %s', str(value))
        self._training_interval = value

    @property
    def trajectory_training_window(self):
        return 1

    @property
    def training_timestep_lag(self):
        return self._training_timestep_lag

    @training_timestep_lag.setter
    def training_timestep_lag(self, value):
        logger.info('Setting value to %s', str(value))
        self._training_timestep_lag = value
