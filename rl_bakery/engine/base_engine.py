from rl_bakery.data_manager.data_manager import DATANAME, MissingDataException
from rl_bakery.engine.dependency_planner import DependencyPlanner
from rl_bakery.operation.train_agent_operation import TrainAgentOperation
from rl_bakery.operation.build_time_step_operation import BuildTimestepOperation
from rl_bakery.operation.operation_factory import OperationFactory
import logging

logger = logging.getLogger(__name__)


class BaseEngine(object):
    """
    This engine will provide the interface which the user needs in order to run the pipeline such
    as init, train and deploy functionality
    """

    # TODO: have the engine composed of an operation planner and runner instead of initializing a new one every time
    # This requires a change in the planner
    def __init__(self, application, dm):
        self._application = application
        self._dm = dm

        available_operators_map = self._get_available_operators()
        self._operation_factory = OperationFactory(available_operators_map, application, dm)

    @classmethod
    def _get_available_operators(cls):
        return {
            TrainAgentOperation.__name__: TrainAgentOperation,
            BuildTimestepOperation.__name__: BuildTimestepOperation
        }

    def init(self, force_run=False):
        """
        initialize the engine. this will fail if the call happens after run 0.
        The caller can set force_run to True to ignore the failure
        """
        if not force_run:
            # make sure that the engine run is less than 1
            run_id = self._application.timing_data.get_current_run_id()
            if run_id >= 1 and not force_run:
                raise Exception("Can't initialize engine since current run_id {%s} > 0" % str(run_id))

        self._init_run_context()
        self._init_agent()

    def train(self, run_id=None, force_run=False):
        """
        execute a training run using data collected up to the provided run_id. If run_id, is not provided,
        get_current_run_id in the engine config will be used to compute it implicitly.

        This function will fail if operation was already executed for the selected run_id. The caller can force the
        operation by setting Force to True

        Params:
            run_id (int): run id
            force_run (boolean): if True, the operation will be executed even if it was run previously using the same
                             run_id
        """
        # compute the current run id if not provided
        if run_id is None:
            run_id = self._application.timing_data.get_current_run_id()

        if not force_run:
            # check if the model was already trained
            try:
                agent = self._dm.get(DATANAME.MODEL, run_id)
                if agent:
                    raise Exception("Train operation was already run successfully for run_id: %s" % str(run_id))
            except MissingDataException:
                pass

        logger.info("Start planing for training")
        ordered_operation_list = self._plan(DATANAME.MODEL, run_id)

        logger.info("Start running operations")
        self._run(ordered_operation_list)

        logger.info("Train command completed.")

    def _init_run_context(self):
        """
        Initialize a dictionary which will hold the run state of the engine historically
        """
        # TODO (idea): set an interface to use the run_context and abstract away it's implementation of the run
        # so that it can be customized (where it is stored and what is it composed of.
        run_context_dict = {"available_data": []}

        # add states tracked by the available operators
        for _, operator in self._get_available_operators().items():
            for state, default_value in operator.tracked_run_state().items():
                run_context_dict[state] = default_value

        run_id = 0
        self._dm.store(run_context_dict, DATANAME.RUN_CONTEXT, run_id)

    def _init_agent(self):

        # init agent
        logger.info("initializing agent")
        agent = self._application.agent.init_agent()

        run_id = 0
        self._dm.store(agent, DATANAME.MODEL, run_id)

        run_context_dict = self._dm.get(DATANAME.RUN_CONTEXT, 0)
        run_context_dict["available_data"].append((DATANAME.MODEL, run_id))
        self._dm.store(run_context_dict, DATANAME.RUN_CONTEXT, run_id)

    def _plan(self, data_name, run_id):
        run_context_dict = self._dm.get_latest(DATANAME.RUN_CONTEXT, run_id)
        available_data = run_context_dict["available_data"]

        # TODO: remove coupling between dependency planner and operator.
        available_operators = self._get_available_operators()

        planner = DependencyPlanner(available_operators, self._application.timing_data, available_data)
        plan = planner.plan(data_name, run_id)

        # TODO: This is debugging code. Only log it if required
        logger.info("Planned Operations:")
        for p in plan:
            logger.info("-ts: %s Op: %s" % (p.run_id, p.op_name))

        return plan

    def _run(self, operation_config_list):

        logger.info("Runner Started")
        for op_config in operation_config_list:
            logger.info("Building Operation: %s" % str(op_config.op_name))
            operation = self._operation_factory.build(op_config.op_name)
            logger.info("Running Operation: %s for run_id: %s" % (str(op_config.op_name), str(op_config.run_id)))
            operation.run(op_config.run_id)
