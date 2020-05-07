from rl_bakery.engine.abstract_engine_config import AbstractEngineConfig


class TFEnvEngineConfig(AbstractEngineConfig):
    def __init__(self, start_dt, training_interval, trajectory_training_window,
                 application_name, version):

        self._first_time_step_dt = start_dt
        self._trajectory_training_window = trajectory_training_window
        self._application_name = application_name
        self._version = version
        self._training_interval = training_interval

    @property
    def application_name(self):
        """Return name of the application"""
        return self._application_name

    @property
    def version(self):
        """Return version of the application (i.e v1)"""
        return self._version

    @property
    def start_dt(self):
        return self._first_time_step_dt

    @property
    def trajectory_training_window(self):
        return self._trajectory_training_window

    @property
    def training_interval(self):
        return self._training_interval
