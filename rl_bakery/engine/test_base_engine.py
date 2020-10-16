from datetime import datetime, timedelta, date
from rl_bakery.applications.tf_rl_application import MockRLApplication
from rl_bakery.data_manager.data_manager import DATANAME
from rl_bakery.engine.dependency_planner import DependencyPlanner
from rl_bakery.engine.base_engine import BaseEngine
from rl_bakery.operation.operation_config import OperationConfig
from rl_bakery.operation.train_agent_operation import TrainAgentOperation
from rl_bakery.operation.operation_factory import OperationFactory
from unittest.mock import call, patch, MagicMock
from tf_agents import specs
from rl_bakery.applications import agent_application
from rl_bakery.agents.abstract import Agent, AgentConfig

import unittest


class TestBaseEngine(unittest.TestCase):

    def setUp(self):

        mock_agent = MagicMock(Agent)
        dataspec = agent_application.DataSpec(
            observation_spec=specs.ArraySpec([1,2,3], int),
            action_spec=specs.ArraySpec([1], float)
        )
        conf = agent_application.make_config(AgentConfig(), [])

        today = datetime(date.today().year, date.today().month, date.today().day)
        env = MagicMock()
        self._mock_agent_init = "MOCKED AGENT"
        mock_agent.init_agent = MagicMock(return_value=self._mock_agent_init)
        self._application = agent_application.AgentApplication(
            data_spec=dataspec,
            agent=mock_agent,
            env=env,
            config=conf,
            first_timestep_dt=today,
            training_interval=timedelta(days=1)
        )


    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    def test_init_success(self, mock_dm):

        run_context = {
            TrainAgentOperation.TRAINING_GLOBAL_STEP: 0,
            "available_data": []
        }
        mock_dm.get.return_value = run_context

        engine = BaseEngine(self._application, mock_dm)
        engine._dm = mock_dm
        engine.init()

        expected_run_context = {
            TrainAgentOperation.TRAINING_GLOBAL_STEP: 0,
            "available_data": [(DATANAME.MODEL, 0)]
        }
        mock_dm.get.assert_any_call(DATANAME.RUN_CONTEXT, 0)

        store_calls = [
            call(self._mock_agent_init, DATANAME.MODEL, 0),
            call(expected_run_context, DATANAME.RUN_CONTEXT, 0)
        ]
        mock_dm.store.assert_has_calls(store_calls, any_order=False)

    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    def test_init_run_id_not_0(self, mock_dm):
        self._application.first_timestep_dt -= timedelta(days=1)
        engine = BaseEngine(self._application, mock_dm)

        with self.assertRaises(Exception):
            engine.init()

    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    @patch('rl_bakery.operation.train_agent_operation.TrainAgentOperation', autospec=True)
    @patch.object(OperationFactory, 'build', autospec=True)
    @patch.object(DependencyPlanner, '__init__', return_value=None)
    @patch.object(DependencyPlanner, 'plan', return_value=None)
    def test_train_implicit_run_id(self, mock_dp_plan, mock_dp_init, mock_factory_build, mock_train_operation, mock_dm):

        run_id = 1
        mock_train_operation.run.return_value = None
        mock_factory_build.return_value = mock_train_operation
        mock_dp_plan.return_value = [OperationConfig(TrainAgentOperation.__name__, run_id)]

        run_context = {
            TrainAgentOperation.TRAINING_GLOBAL_STEP: 0,
            "available_data": [("model", 0)]
        }

        mock_dm.get_latest.return_value = run_context
        mock_dm.get.return_value = False

        self._application.first_timestep_dt -= timedelta(days=1)
        engine = BaseEngine(self._application, mock_dm)
        engine.train()

        mock_dm.get.assert_any_call(DATANAME.MODEL, run_id)
        mock_dm.get_latest.assert_any_call(DATANAME.RUN_CONTEXT, run_id)
        mock_dp_plan.assert_any_call(DATANAME.MODEL, run_id)
        mock_train_operation.run.assert_any_call(run_id)

    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    @patch('rl_bakery.operation.train_agent_operation.TrainAgentOperation', autospec=True)
    @patch.object(OperationFactory, 'build', autospec=True)
    @patch.object(DependencyPlanner, '__init__', return_value=None)
    @patch.object(DependencyPlanner, 'plan', return_value=None)
    def test_train_explicit_run_id(self, mock_dp_plan, mock_dp_init, mock_factory_build, mock_train_operation, mock_dm):

        metadata = {
            TrainAgentOperation.TRAINING_GLOBAL_STEP: 0,
            "available_data": [("model", 0)]
        }
        mock_dm.get_latest.return_value = metadata
        mock_dm.get.return_value = False

        mock_train_operation.run.return_value = None
        mock_factory_build.return_value = mock_train_operation

        run_id = 1
        mock_dp_plan.return_value = [OperationConfig(TrainAgentOperation.__name__, run_id)]

        engine = BaseEngine(self._application, mock_dm)
        engine.train(run_id)

        mock_dm.get.assert_any_call(DATANAME.MODEL, run_id)
        mock_dm.get_latest.assert_any_call(DATANAME.RUN_CONTEXT, run_id)
        mock_dp_plan.assert_any_call(DATANAME.MODEL, run_id)
        mock_train_operation.run.assert_any_call(run_id)

    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    def test_train_agent_exist(self, mock_dm):

        run_id = 1
        mock_dm.get.return_value = True
        engine = BaseEngine(self._application, mock_dm)
        engine._dm = mock_dm

        with self.assertRaises(Exception):
            engine.train(run_id)

    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    @patch('rl_bakery.operation.train_agent_operation.TrainAgentOperation', autospec=True)
    @patch.object(OperationFactory, 'build', autospec=True)
    @patch.object(DependencyPlanner, '__init__', return_value=None)
    @patch.object(DependencyPlanner, 'plan', return_value=None)
    def test_train_agent_exist_force(self, mock_dp_plan, mock_dp_init, mock_factory_build, mock_train_operation,
                                     mock_dm):

        metadata = {
            TrainAgentOperation.TRAINING_GLOBAL_STEP: 0,
            "available_data": [("model", 0)]
        }
        mock_dm.get_latest.return_value = metadata
        mock_dm.get.return_value = True

        mock_train_operation.run.return_value = None
        mock_factory_build.return_value = mock_train_operation

        run_id = 1
        mock_dp_plan.return_value = [OperationConfig(TrainAgentOperation.__name__, run_id)]

        engine = BaseEngine(self._application, mock_dm)
        engine._dm = mock_dm
        engine.train(run_id, True)

        mock_dm.get_latest.assert_any_call(DATANAME.RUN_CONTEXT, run_id)
        mock_dp_plan.assert_any_call(DATANAME.MODEL, run_id)
        mock_train_operation.run.assert_any_call(run_id)
