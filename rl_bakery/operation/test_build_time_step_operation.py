from datetime import datetime, timedelta
from rl_bakery.engine.timing_data import TimingData
from rl_bakery.data_manager.data_manager import DATANAME
from rl_bakery.operation.build_time_step_operation import BuildTimestepOperation
from rl_bakery.spark_utilities import PySparkTestCase
from unittest import TestCase
from unittest.mock import call, patch
from omegaconf import OmegaConf
import mock

class MockEnv:

    def build_time_steps(self, previous_run_dt, current_run_dt):
        pass

class BuildTimestepOperationTest(PySparkTestCase, TestCase):
    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    def test_timestep_operation(self, mock_data_manager):
        start_dt = datetime.now()
        training_interval = timedelta(days=1)

        mock_env = mock.MagicMock()

        conf = OmegaConf.from_dotlist(["project.tensorboard_path=/tmp/test_tb/"])
        mock_app = mock.MagicMock()
        mock_app.timing_data = TimingData(start_dt=start_dt, training_interval=training_interval)
        mock_app.env = mock_env
        mock_app.config = conf
        mock_env.env_id_cols = ["env_id_1", "env_id_2"]
        mock_env.ts_id_col = "ts_1"
        mock_env.obs_cols = ["obs_1", "obs_2"]

        mock_timestep = [{"env_id_1": 1, "env_id_2": 2, "ts_1": 1, "discount": 1.0, "obs_1": 1, "obs_2": 2, "action": 1, "reward": 0.0,
                          "step_type": 0}]
        mock_timestep_df = self.spark.createDataFrame(mock_timestep)

        metadata_dict = {
                "available_data": [("test_data", 0)]
            }

        mock_env.build_time_steps = mock.MagicMock(return_value=mock_timestep_df)
        mock_data_manager.get_latest.return_value = metadata_dict

        run_id = 1
        operation = BuildTimestepOperation(mock_app, mock_data_manager)
        operation.run(run_id)

        mock_data_manager.get_latest.assert_any_call(DATANAME.RUN_CONTEXT, run_id)

        expected_start_dt = start_dt
        expected_end_dt = start_dt + training_interval
        mock_env.build_time_steps.assert_called_with(expected_start_dt, expected_end_dt)

        expected_metadata = {
            "available_data": [
                ("test_data", 0),
                (DATANAME.TIMESTEP, run_id)
            ]
        }

        calls = [
            call(mock_timestep_df, DATANAME.TIMESTEP, run_id),
            call(expected_metadata, DATANAME.RUN_CONTEXT, run_id)]
        mock_data_manager.store.assert_has_calls(calls, any_order=False)
