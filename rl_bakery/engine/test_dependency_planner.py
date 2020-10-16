from unittest import TestCase
from datetime import date, datetime, timedelta
from rl_bakery.engine.dependency_planner import DependencyPlanner, OperationConfig
from rl_bakery.engine.timing_data import TimingData


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
        today = datetime(date.today().year, date.today().month, date.today().day)
        timing_data = TimingData(
            start_dt=today,
            training_interval=timedelta(days=1)
        )
        available_data = []
        planner = DependencyPlanner(operation_list, timing_data, available_data)

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
        available_data = [(Operation2.output_dataname(), 0), (Operation1.output_dataname(), 1)]
        today = datetime(date.today().year, date.today().month, date.today().day)
        timing_data = TimingData(
            start_dt=today,
            training_interval=timedelta(days=1)
        )
        planner = DependencyPlanner(operation_list, timing_data, available_data)

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
        available_data = [(Operation2.output_dataname(), 0)]
        today = datetime(date.today().year, date.today().month, date.today().day)
        timing_data = TimingData(
            start_dt=today,
            training_interval=timedelta(days=1)
        )
        planner = DependencyPlanner(operation_list, timing_data, available_data)

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
        available_data = [(Operation2.output_dataname(), 0), (Operation1.output_dataname(), 1)]
        today = datetime(date.today().year, date.today().month, date.today().day)
        timing_data = TimingData(
            start_dt=today,
            training_interval=timedelta(days=1)
        )
        planner = DependencyPlanner(operation_list, timing_data, available_data)

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
        available_data = []
        today = datetime(date.today().year, date.today().month, date.today().day)
        timing_data = TimingData(
            start_dt=today,
            training_interval=timedelta(days=1)
        )
        planner = DependencyPlanner(operation_list, timing_data, available_data)

        with self.assertRaises(Exception):
            planner.plan(Operation2.output_dataname(), 2)
