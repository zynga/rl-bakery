from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class TimingData:
    start_dt: datetime
    training_interval: timedelta
    trajectory_training_window: int = 1
    training_timestep_lag: int = 1


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
