from rl_bakery.operation.base_operation import BaseOperation
from rl_bakery.data_manager.data_manager import DATANAME
import logging

logger = logging.getLogger(__name__)


class BuildTimestepOperation(BaseOperation):
    """
    This operator retrieves new Timesteps based on the interactions between the agent and the environment since
    it's last run. By performing this incrementally, only new timesteps are needed.

    The result is a parquet file that contains several timesteps and will be persisted in S3.

    A Timestep is composed of:
        * <env_id>: a set of columns that uniquely identify the environment the agent is interacting
                    with i.e. [player_id, match_id]
        * timestep_id: a column that allows to uniquely identify and order a timestep within a series of
        interactions between the agent and an environment. i.e. move_id or move_timetamp
        * <observation features>: a list of columns of any type
        * reward: one column of type float. column name must be 'reward'
        * action: one column of type string. column name must be 'action'
        * step_type: one categorical column of type string. Content has to be one of these: "first", "last", "middle".
                    column name must be 'step_type'
        * policy_info (optional): this will contains metadata in relation to the selected action
    """

    REQUIRED_COLUMNS = ["action", "reward", "step_type", "discount"]

    @classmethod
    def output_dataname(cls):
        return DATANAME.TIMESTEP

    def _run(self, run_id, rl_app, engine_config):
        """
        Build new timesteps by calling the RL application delegate. Validate it before storing it.

        Params:
            run_id: the id of the training run
            run_dt: the datetime of the training run
            metadata: metadata information for this run

        Returns:
            A Spark dataframe containing timesteps
        """

        # compute time window
        start_dt, end_dt = self._compute_time_window(engine_config.start_dt, engine_config.training_interval, run_id)

        # call rl app to build timesteps
        logger.info("build_time_step_operation run_id={} start_dt={} end_dt={}".format(run_id, start_dt, end_dt))
        timestep_df = rl_app.build_time_steps(start_dt, end_dt)

        self._validate_timestep(timestep_df)

        reward_list = timestep_df.select("reward").collect()
        reward_list = [rwd["reward"] for rwd in reward_list]
        self.log_to_tensorboard(metric_name='rewards', data=reward_list)
        self.log_to_tensorboard(metric_name='avg_rewards', data=sum(reward_list) / len(reward_list))

        action_list = timestep_df.select("action").collect()
        action_list = [act["action"] for act in action_list]
        self.log_to_tensorboard(metric_name='actions', data=action_list)

        step_type_list = timestep_df.select("step_type").collect()
        step_type_list = [step_type["step_type"] for step_type in step_type_list]
        self.log_to_tensorboard(metric_name='step_type', data=step_type_list)

        return timestep_df

    def _validate_timestep(self, timestep_df):
        """
        Make sure that all required columns are present
        """
        rl_app = self._rl_app
        # df contains required columns
        required_cols = self.REQUIRED_COLUMNS + rl_app.env_id_cols + [rl_app.ts_id_col] + rl_app.obs_cols
        for col in required_cols:
            if col not in timestep_df.columns:
                raise Exception("Missing column from timestep_df: %s. Got: %s" %
                                (str(col), str(timestep_df.columns)))

    @staticmethod
    def _compute_time_window(start_date, training_interval, run_id):
        """
        compute the run time window given the start date of the application and the current run_id

        Return:
            pair of datetime containing start and end boundaries of the start and end of the period
        """

        end_dt = start_date + run_id * training_interval
        start_dt = end_dt - training_interval

        return start_dt, end_dt


def _get_column_type(df, col_name):
    return df.select(col_name).dtypes[0][1]
