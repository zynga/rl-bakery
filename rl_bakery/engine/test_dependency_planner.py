from unittest import TestCase
from rl_bakery.engine.abstract_engine_config import MockEngineConfig
from rl_bakery.engine.dependency_planner import DependencyPlanner, OperationConfig


class Operation1(object):
    @staticmethod
    def output_dataname():
        return "data_1"

    @staticmethod
    def data_dependencies(_):
        return {}


class Operation2(object):
    @staticmethod
    def output_dataname():
        return "data_2"

    @staticmethod
    def data_dependencies(_):
        return {
            "datapoint_1": (Operation1.output_dataname(), 0),
            "datapoint_2": (Operation2.output_dataname(), -1)}


class TestDependencyPlanner(TestCase):

    def test_no_dependencies(self):
        operation_list = {
            "op_1": Operation1,
            "op_2": Operation2
        }
        mock_rl_engine = MockEngineConfig()
        available_data = []
        planner = DependencyPlanner(operation_list, mock_rl_engine, available_data)

        actual_plan = planner.plan(Operation1.output_dataname(), 2)
        expected_plan = [
            OperationConfig(Operation1.__name__, 2)
        ]
        self.assertEquals(expected_plan, actual_plan)

    def test_satisfied_dependency(self):
        operation_list = {
            "op_1": Operation1,
            "op_2": Operation2
        }
        mock_rl_engine = MockEngineConfig()
        available_data = [(Operation2.output_dataname(), 0), (Operation1.output_dataname(), 1)]
        planner = DependencyPlanner(operation_list, mock_rl_engine, available_data)

        actual_plan = planner.plan(Operation2.output_dataname(), 1)
        expected_plan = [
            OperationConfig(Operation2.__name__, 1)
        ]
        self.assertEquals(expected_plan, actual_plan)

    def test_back_filling_multiple_runs(self):
        operation_list = {
            "op_1": Operation1,
            "op_2": Operation2
        }
        mock_rl_engine = MockEngineConfig()
        available_data = [(Operation2.output_dataname(), 0)]
        planner = DependencyPlanner(operation_list, mock_rl_engine, available_data)

        actual_plan = planner.plan(Operation2.output_dataname(), 2)
        expected_plan = [
            OperationConfig(Operation1.__name__, 2),
            OperationConfig(Operation1.__name__, 1),
            OperationConfig(Operation2.__name__, 1),
            OperationConfig(Operation2.__name__, 2),
        ]
        self.assertEquals(expected_plan, actual_plan)

    def test_back_filling_multiple_runs_and_skip(self):
        operation_list = {
            "op_1": Operation1,
            "op_2": Operation2
        }
        mock_rl_engine = MockEngineConfig()
        available_data = [(Operation2.output_dataname(), 0), (Operation1.output_dataname(), 1)]
        planner = DependencyPlanner(operation_list, mock_rl_engine, available_data)

        actual_plan = planner.plan(Operation2.output_dataname(), 2)
        expected_plan = [
            OperationConfig(Operation1.__name__, 2),
            OperationConfig(Operation2.__name__, 1),
            OperationConfig(Operation2.__name__, 2),
        ]
        self.assertEquals(expected_plan, actual_plan)

    def test_error(self):
        operation_list = {
            "op_1": Operation1,
            "op_2": Operation2
        }
        mock_rl_engine = MockEngineConfig()
        available_data = []
        planner = DependencyPlanner(operation_list, mock_rl_engine, available_data)

        with self.assertRaises(Exception):
            planner.plan(Operation2.output_dataname(), 2)
