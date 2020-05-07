from datetime import datetime, timedelta
from rl_bakery.data_manager.data_manager import DATANAME
from rl_bakery.engine.abstract_engine_config import MockEngineConfig
from rl_bakery.operation.build_time_step_operation import BuildTimestepOperation
from rl_bakery.spark_utilities import PySparkTestCase
from unittest import TestCase
from unittest.mock import call, patch


class BuildTimestepOperationTest(PySparkTestCase, TestCase):
    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    @patch('rl_bakery.applications.abstract_rl_application.AbstractRLApplication', autospec=True)
    def test_timestep_operation(self, mock_abstract_app, mock_data_manager):
        start_dt = datetime.now()
        training_interval = timedelta(days=1)

        mock_engine_config = MockEngineConfig()
        mock_engine_config.start_dt = start_dt
        mock_engine_config.training_interval = training_interval

        mock_timestep = [{"env_id_1": 1, "env_id_2": 2, "ts_1": 1, "obs_1": 1, "obs_2": 2, "action": 1, "reward": 0.0,
                          "step_type": 0}]
        mock_timestep_df = self.spark.createDataFrame(mock_timestep)

        metadata_dict = {
                "available_data": [("test_data", 0)]
            }

        mock_abstract_app.build_time_steps.return_value = mock_timestep_df
        mock_data_manager.get_latest.return_value = metadata_dict

        run_id = 1
        operation = BuildTimestepOperation(mock_abstract_app, mock_engine_config, mock_data_manager)
        operation.run(run_id)

        mock_data_manager.get_latest.assert_any_call(DATANAME.RUN_CONTEXT, run_id)

        expected_start_dt = start_dt
        expected_end_dt = start_dt + training_interval
        mock_abstract_app.build_time_steps.assert_called_with(expected_start_dt, expected_end_dt)

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
