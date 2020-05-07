from rl_bakery.applications.tf_rl_application import MockRLApplication
from rl_bakery.data_manager.data_manager import DATANAME
from rl_bakery.engine.abstract_engine_config import MockEngineConfig
from rl_bakery.operation.train_agent_operation import TrainAgentOperation
from rl_bakery.replay_buffer.replay_buffer_abstract import ReplayBufferAbstract
from rl_bakery.spark_utilities import PySparkTestCase

from unittest import TestCase
from unittest.mock import call, patch


class MockReplayBuffer(ReplayBufferAbstract):

    def add_batch(self, traj_dict):
        pass

    def get_batch(self, mini_batch_size):
        pass


class TestTrainActionOperation(PySparkTestCase, TestCase):
    @patch('rl_bakery.agent_abstract.MockAgent', autospec=True)
    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    @patch('rl_bakery.operation.trajectory_builder.TrajectoryBuilder', autospec=True)
    @patch('rl_bakery.operation.test_train_agent_operation.MockReplayBuffer', autospec=True)
    def test_run(self, mock_rb, mock_tb, mock_data_manager, mock_agent):

        mock_engine_config = MockEngineConfig()
        mock_rl_app = MockRLApplication()
        MockRLApplication._env_id_cols = ["env_id_1"]
        MockRLApplication._ts_id_col = "ts_1"

        mock_rl_app.agent_config = {
            "num_iterations": 1,
            "mini_batch_size": 32
        }

        run_context_dict = {
            "available_data": [("test_data", 0)],
            TrainAgentOperation.TRAINING_GLOBAL_STEP: 0
        }
        mock_data_manager.get_latest.return_value = run_context_dict

        mock_timestep = [{"env_id_1": 1, "env_id_2": 2, "ts_1": 1, "obs_1": 1, "obs_2": 2,
                          "action": 1, "reward": 0.0, "step_type": 0}]
        mock_timestep_df = self.spark.createDataFrame(mock_timestep)

        def get_side_effect(data_name, _):
            if data_name == DATANAME.TIMESTEP:
                return mock_timestep_df
            else:
                return mock_agent

        mock_data_manager.get.side_effect = get_side_effect
        fake_mini_batch = "fake_mini_batch"

        class MockMeta(object):
            def __init__(self, prob):
                self.probabilities = prob

        fake_meta = MockMeta(0.1)
        mock_rb.get_batch.return_value = fake_mini_batch, fake_meta
        mock_traj_dict = {"observations": [1, 2, 3]}
        mock_tb.run.return_value = mock_traj_dict

        class MockLoss(object):
            def __init__(self, loss):
                self.loss = loss
        mock_loss = MockLoss("mock_loss")
        mock_agent.train.return_value = mock_loss

        run_id = 5
        operation = TrainAgentOperation(mock_rl_app, mock_engine_config, mock_data_manager)
        operation._trajectory_builder = mock_tb
        operation._replay_buffer = mock_rb
        operation.run(run_id)

        get_calls = [
            call(DATANAME.MODEL, run_id - 1),
            call(DATANAME.TIMESTEP, run_id)
        ]
        mock_data_manager.get.assert_has_calls(get_calls, any_order=True)

        mock_tb.run.assert_called_with(mock_timestep_df.collect())
        mock_rb.add_batch.assert_called_with(mock_traj_dict)
        mock_rb.pre_process.assert_called_with(0)
        mock_rb.get_batch.assert_called_with(mock_rl_app.training_config["mini_batch_size"])
        mock_agent.train.assert_called_with(fake_mini_batch, fake_meta.probabilities)
        mock_rb.post_process.assert_called_with(fake_meta, mock_loss, 0)

        expected_metadata = {
            "available_data": [("test_data", 0), (DATANAME.MODEL, run_id)],
            TrainAgentOperation.TRAINING_GLOBAL_STEP: 0 + mock_rl_app.training_config["num_iterations"]
        }
        store_calls = [
            call(mock_agent, DATANAME.MODEL, run_id),
            call(expected_metadata, DATANAME.RUN_CONTEXT, run_id)
        ]
        mock_data_manager.store.assert_has_calls(store_calls, any_order=True)

    @patch('rl_bakery.agent_abstract.MockAgent', autospec=True)
    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    @patch('rl_bakery.operation.trajectory_builder.TrajectoryBuilder', autospec=True)
    @patch('rl_bakery.operation.test_train_agent_operation.MockReplayBuffer', autospec=True)
    def test_run_lag(self, mock_rb, mock_tb, mock_data_manager, mock_agent):

        mock_engine_config = MockEngineConfig()
        mock_engine_config.training_timestep_lag = 2
        mock_rl_app = MockRLApplication()
        MockRLApplication._env_id_cols = ["env_id_1"]
        MockRLApplication._ts_id_col = "ts_1"

        mock_rl_app.agent_config = {
            "num_iterations": 1,
            "mini_batch_size": 32
        }

        run_context_dict = {
            "available_data": [("test_data", 0)],
            TrainAgentOperation.TRAINING_GLOBAL_STEP: 0
        }
        mock_data_manager.get_latest.return_value = run_context_dict

        mock_timestep = [{"env_id_1": 1, "env_id_2": 2, "ts_1": 1, "obs_1": 1, "obs_2": 2,
                          "action": 1, "reward": 0.0, "step_type": 0}]
        mock_timestep_df = self.spark.createDataFrame(mock_timestep)

        def get_side_effect(data_name, _):
            if data_name == DATANAME.TIMESTEP:
                return mock_timestep_df
            else:
                return mock_agent

        mock_data_manager.get.side_effect = get_side_effect
        fake_mini_batch = "fake_mini_batch"

        class MockMeta(object):
            def __init__(self, prob):
                self.probabilities = prob

        fake_meta = MockMeta(0.1)
        mock_rb.get_batch.return_value = fake_mini_batch, fake_meta
        mock_traj_dict = {"observations": [1, 2, 3]}
        mock_tb.run.return_value = mock_traj_dict

        class MockLoss(object):
            def __init__(self, loss):
                self.loss = loss

        mock_loss = MockLoss("mock_loss")
        mock_agent.train.return_value = mock_loss

        run_id = 5
        operation = TrainAgentOperation(mock_rl_app, mock_engine_config, mock_data_manager)
        operation._trajectory_builder = mock_tb
        operation._replay_buffer = mock_rb
        operation.run(run_id)

        get_calls = [
            call(DATANAME.MODEL, run_id - 1),
            call(DATANAME.TIMESTEP, run_id - 2)
        ]
        mock_data_manager.get.assert_has_calls(get_calls, any_order=True)

        mock_tb.run.assert_called_with(mock_timestep_df.collect())
        mock_rb.add_batch.assert_called_with(mock_traj_dict)
        mock_rb.pre_process.assert_called_with(0)
        mock_rb.get_batch.assert_called_with(mock_rl_app.training_config["mini_batch_size"])
        mock_agent.train.assert_called_with(fake_mini_batch, fake_meta.probabilities)
        mock_rb.post_process.assert_called_with(fake_meta, mock_loss, 0)

        expected_metadata = {
            "available_data": [("test_data", 0), (DATANAME.MODEL, run_id)],
            TrainAgentOperation.TRAINING_GLOBAL_STEP: 0 + mock_rl_app.training_config["num_iterations"]
        }
        store_calls = [
            call(mock_agent, DATANAME.MODEL, run_id),
            call(expected_metadata, DATANAME.RUN_CONTEXT, run_id)
        ]
        mock_data_manager.store.assert_has_calls(store_calls, any_order=True)
