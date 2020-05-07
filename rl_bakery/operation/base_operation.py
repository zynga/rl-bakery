from rl_bakery.data_manager.data_manager import DATANAME

import abc
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class BaseOperation(object):
    """
    Operations are executed to output data for the RL pipeline.
    This class is designed so that each Operation class specifies its
    data dependencies required for processing and data outputs.

    The BaseOperator is then responsible for retrieving required data
    before processing and storing the output, allowing Operation
    implementations to focus on data processing.
    """

    def __init__(self, rl_app, engine_config, dm):
        """
        init a pipeline operation.

        Params:
            rl_app: an instance of RLApplication
            engine_config: an instance of EngineConfig
            dm: a data manager
        """
        self._rl_app = rl_app
        self._engine_config = engine_config
        self.__dm = dm

    @classmethod
    def output_dataname(cls):
        """Returns a DATANAME that the operation creates for a given timestep"""
        pass

    @classmethod
    def tracked_run_state(cls):
        """Returns a dict containing the metrics that the operator needs to track over different runs with
        it's value in the initial state"""
        return {}

    def _data_manager(self):
        return self.__dm

    @classmethod
    def data_dependencies(cls, engine_config):
        """A dictionary of variable names to data dependencies, triplets of (DATANAME, offset).
        'offset' is relative to the given timestep to run the operation."""
        return {}

    @classmethod
    def optional_data_dependencies(cls, engine_config):
        """A dictionary of variable names to optional data dependencies that are loaded if possible. Data is
        represented as triplets of (DATANAME, offset). 'offset' is relative to the given timestep to run the
        operation. """
        return {}

    def run(self, run_id):
        """
        Executes the Operation. All dependent resources must be available in the DataManager. After execution,
        the 'output_resources' will be available in DataManager.

        Params:
            run_id: The id of the pipeline run
        """

        tb_counter = self._get_tensorboard_counter(run_id)
        tb_writer = start_tensorboard_writer(self._engine_config.tensorboard_path, tb_counter)
        logger.info("Building %s and run_id: %s" % (self.output_dataname(), str(run_id)))

        required_data_dependencies = self.data_dependencies(self._engine_config)
        logger.info("Required dependencies: %s" % str(required_data_dependencies))
        data, exception_list = self._get_data_dependencies(required_data_dependencies, run_id)
        for e in exception_list:
            logger.warning(str(e))

        if exception_list:
            raise Exception("Critical Error: Failed to load %s data dependencies. more details above." %
                            str(len(exception_list)))

        optional_data_dependencies = self.optional_data_dependencies(self._engine_config)
        optional_data, exception_list = self._get_data_dependencies(optional_data_dependencies, run_id)
        if exception_list:
            for e in exception_list:
                logger.warning(str(e))
            logger.warning("Warning: Failed to load %s data dependencies. more details above." %
                           str(len(exception_list)))
        with tb_writer.as_default():
            output = self._run(run_id, self._rl_app, self._engine_config, **data, **optional_data)

        # TODO: The check should be based on the output_dataname. If it is not set, skip storing the output
        if output:
            self.__dm.store(output, self.output_dataname(), run_id)

        run_state = self.get_run_state(run_id, ["available_data"])
        run_state["available_data"].append((self.output_dataname(), run_id))
        self.update_run_state(run_id, run_state)

        close_tensorboard_writer(tb_writer)

    def _get_tensorboard_counter(self, run_id):
        return run_id

    @abc.abstractmethod
    def _run(self, run_id, rl_app, engine_config, **input_data):
        """
        This function must contain the implementation of the logic specific to the concrete operator

        Params:
            run_id: the run id
            rl_app: the RL application config
            engine_config: the RL engine config
            **input_data: any other dependency loaded from the data manager and specific to the concrete operator
        """

    def update_run_state(self, run_id, new_run_state_dict):
        """
        store the new values provided in run_state dict in run context in data manager

        Params:
            run_id: the run_id in which metric is stored
            new_run_state_dict: A dictionary mapping run_state keys to their new values
        """
        logger.info("updating run_context with %s" % (str(new_run_state_dict)))
        run_context_dict = self.__dm.get_latest(DATANAME.RUN_CONTEXT, run_id)
        for k, v in new_run_state_dict.items():
            run_context_dict[k] = v
        logger.info("run_context updated. content: %s" % str(run_context_dict))
        self.__dm.store(run_context_dict, DATANAME.RUN_CONTEXT, run_id)

    def get_run_state(self, run_id, run_state_keys):
        """
        retrieve the most recently stored values for the run state keys provided

        Params:
            run_id: the run_id that the run context will be loaded from
            run_state_keys: a list of strings of the run state keys that need to be retrieved

        Return:
            a dict containing the run state values of the requested keys
        """
        run_context_dict = self.__dm.get_latest(DATANAME.RUN_CONTEXT, run_id)
        res = {k: run_context_dict.get(k) for k in run_state_keys}

        return res

    def _get_data_dependencies(self, data_dependencies_dict, run_id):
        out = {}
        exception_list = []

        for var, (data_name, offset) in data_dependencies_dict.items():
            selected_run_id = run_id + offset
            try:
                out[var] = self.__dm.get(data_name, selected_run_id)
            except Exception as e:
                exception_list.append(e)

        return out, exception_list

    # TODO: we could create an logging interface that make logging generic
    @staticmethod
    def log_to_tensorboard(metric_name, data):
        if type(data) is list:
            tf.compat.v2.summary.histogram(name=metric_name, data=data)
        else:
            logger.info("%s: %s" % (metric_name, str(data)))
            tf.compat.v2.summary.scalar(name=metric_name, data=data)


def start_tensorboard_writer(tb_path, global_step_number, flush_millis=200):
    tensorboard_log_dir = tb_path
    logger.info("Setting up tensorboard now")
    logger.info("tensorboard_log_dir: %s" % str(tensorboard_log_dir))
    logger.info("global_step_number: %s" % str(global_step_number))

    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf.compat.v2.summary.experimental.set_step(global_step)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.assign(global_step, global_step_number)
    logger.info("Global step set to: %s" % str(global_step_number))

    tb_writer = tf.compat.v2.summary.create_file_writer(tensorboard_log_dir, flush_millis=flush_millis)
    tb_writer.set_as_default()
    logger.info("Tensorboard is now On")

    return tb_writer


def close_tensorboard_writer(tb_writer):
    tb_writer.close()
