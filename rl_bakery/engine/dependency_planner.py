from rl_bakery.operation.operation_config import OperationConfig
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DependencyPlanner:
    """Creates a plan of OperationConfigs to run in order to generate a requested data item for a given set of
    available data items in the DataManager."""

    # TODO (idea): remove coupling between dependency planner and operator. Instead of take a map of operators,
    # take in a graph between data dependencies and operator-data mapping
    # TODO: It makes more sense not setting available_data as an attribute but pass it when plan() is called since
    # available data changes very often
    def __init__(self, operations_map, timing_data, available_data=None):
        """
        :param operations: A List of Operation classes that the plan can execute
        :param timing_data: Information about timing for the application
        :param available_data: A List of (DATANAME, timestep) tuples of data already available
        """

        # dict data to ops to execute after data is complete
        # todo: this can be created in 1 line
        self._data_to_operation = {}
        for _, o in operations_map.items():
            d = o.output_dataname()
            # TODO: Move this outside of the this loop. Place it inside of a planner validation step
            if d:
                if d in self._data_to_operation:
                    raise Exception("Duplicate operation found to generate %s" % d)
                self._data_to_operation[d] = o

        self._available_data = available_data
        self._timing_data = timing_data

    # TODO: add support for operations that do not output data like DeployOperation.
    def plan(self, data_name, run_id):

        logger.info("Planning for dataname: %s, run_id: %s" % (str(data_name), str(run_id)))
        ops_and_run_ids = self._get_ops_to_generate_data(data_name, run_id)
        plan = [self._make_op_config(op, r_id) for (op, r_id) in ops_and_run_ids]

        return plan

    def _get_ops_to_generate_data(self, data_name, run_id):
        # Return topological sort to build dependencies before requested data
        # One way to get a topological sort is reversing post order depth-first-search

        # Get all ops required to be run to generate requested data item by traversing all dependencies back to
        # available data
        # TODO: Why are we passing the _get_op_dependencies function?
        required_ops = self._postorder_dfs(data_name, run_id, self._get_op_dependencies)

        # Reverse dependencies edges to
        op_to_outputs = self._get_op_to_downstream_data(required_ops)

        ops_and_timesteps = []

        # Select a data/timestep from the required ops that hasn't yet been visited
        finished = {}
        visited = {}
        for (op, op_ts) in required_ops:
            op_output_data = op.output_dataname()
            if (op_output_data, op_ts) not in ops_and_timesteps:
                dfs_out = self._postorder_dfs(op_output_data, op_ts, lambda op, ts: op_to_outputs[(op, ts)], visited,
                                              finished, None)
                ops_and_timesteps += dfs_out

        ops_and_timesteps.reverse()

        return ops_and_timesteps

    def _get_op_dependencies(self, op, run_id):
        values = op.data_dependencies(self._timing_data).values()
        return [(dep_data_name, offset + run_id) for (dep_data_name, offset) in values]

    def _get_op_to_downstream_data(self, ops):
        """Returns a map of each (op, timestep) in ops to the list of (dataname, timestep) dependent on it"""
        downstream_ops = {(op, r_id): [] for (op, r_id) in ops}
        for (op, r_id) in ops:
            for (dep_data_name, dep_run_id) in self._get_op_dependencies(op, r_id):
                if (dep_data_name, dep_run_id) not in self._available_data:
                    dep_op = self._data_to_operation[dep_data_name]
                    downstream_ops[(dep_op, dep_run_id)].append((op.output_dataname(), r_id))

        return downstream_ops

    def _postorder_dfs(self, data_name, run_id, get_op_data, visited=None, finished=None, parent=None):
        if visited is None:
            visited = {}
        if finished is None:
            finished = {}
        if (data_name, run_id) in finished or run_id < 0:
            # This dependency has already been added to the plan
            return []

        if (data_name, run_id) in visited:
            raise Exception("Cyclical dependency loop found, cannot make plan.")

        # TODO: what happened to temporary vs permanent markers? what happens if 2
        visited[data_name, run_id] = True
        op = self._data_to_operation[data_name]

        traversal = []
        for (dep_data_name, dep_run_id) in get_op_data(op, run_id):
            if not self._skip_generating_data(dep_data_name, dep_run_id):
                traversal = self._postorder_dfs(dep_data_name, dep_run_id, get_op_data, visited, finished,
                                                (data_name, run_id)) + traversal

        finished[(data_name, run_id)] = True
        return traversal + [(op, run_id)]

    def _make_op_config(self, operation, run_Id):
        return OperationConfig(operation.__name__, run_Id)

    def _skip_generating_data(self, dataname, run_id):
        return (dataname, run_id) in self._available_data
