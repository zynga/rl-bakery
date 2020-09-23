from rl_bakery.data_manager.data_manager import DATANAME
from rl_bakery.operation.base_operation import BaseOperation
from rl_bakery.operation.trajectory_builder import TrajectoryBuilder

import logging
import tensorflow as tf
import time

logger = logging.getLogger(__name__)


class TrainAgentOperation(BaseOperation):

    TRAINING_GLOBAL_STEP = "training_global_step"

    @classmethod
    def output_dataname(cls):
        return DATANAME.MODEL

    @classmethod
    def data_dependencies(cls, timing_data):
        return {
            "agent": (DATANAME.MODEL, -1),
            "timestep_df": (DATANAME.TIMESTEP, -timing_data.training_timestep_lag),
        }

    @classmethod
    def optional_data_dependencies(cls, timing_data):
        # get the window size from which trajectories will be loaded
        trajectory_training_window = timing_data.trajectory_training_window

        # compute the number training runs to use excluding the current one
        num_previous_training_runs = trajectory_training_window - 1

        # retrieve previous trajectories other than the one from the current timestep.These trajectories must be
        # in the time window specified in the app
        result = {}
        for i in range(1, num_previous_training_runs):
            offset = -i
            result["timestep_offset_%s_df" % str(offset)] = (DATANAME.TIMESTEP,
                                                             offset - timing_data.training_timestep_lag)

        return result

    @classmethod
    def tracked_run_state(cls):
        """Returns a dict containing the metrics that the operator needs to track over different runs with
        it's value in the initial state"""
        return {TrainAgentOperation.TRAINING_GLOBAL_STEP: 0}

    def __init__(self, application, dm):
        super().__init__(application, dm)

        # init a trajectory builder
        env_id_cols = application.env.env_id_cols
        ts_id_col = application.env.ts_id_col
        obs_cols = application.env.obs_cols
        n_step = application.config.trajectory.n_step
        self._trajectory_builder = TrajectoryBuilder(obs_cols, env_id_cols, ts_id_col, n_step)

        self._replay_buffer = self._application.init_replay_buffer()

    def _get_tensorboard_counter(self, run_id):
        run_state = self.get_run_state(run_id, [self.TRAINING_GLOBAL_STEP])
        training_global_step = run_state[self.TRAINING_GLOBAL_STEP]

        return training_global_step

    def _run(self, run_id, agent, timestep_df, **previous_timestep_dict):
        """
        This updates the given agent with the latest timesteps.

        :param run_id: The run id to train for
        :param agent: The agent to update
        :param timestep_df: A Spark dataframe of the latest timesteps
        :param previous_timestep_dict: Any previously available timesteps (these will be added to the latest)
        :return: the given agent is updated and returned
        """

        logger.info("Starting training for run_sid %s" % str(run_id))

        # build a trajectory based on the provided timesteps
        for _, ts_df in previous_timestep_dict.items():
            timestep_df = timestep_df.union(ts_df)

        # TODO: move this order and trajectory building to it's own operation so that train can
        # run on gpu while trajectory building run on spark
        timestep_df = timestep_df.orderBy(*self._application.env.env_id_cols, self._application.env.ts_id_col)
        traj_dict = self._trajectory_builder.run(timestep_df.collect())

        # setup replay buffer
        self._replay_buffer.add_batch(traj_dict)

        # train the agent
        global_step = tf.compat.v1.train.get_or_create_global_step()
        summary_interval = self._application.config.project.summary_interval
        with tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
            num_iterations = self._application.config.training.num_iterations
            mini_batch_size = self._application.config.training.batch_size
            self._train(agent, self._replay_buffer, num_iterations, mini_batch_size)

        # update global step in run_context so that it can be used in the next run
        run_state = self.get_run_state(run_id, [self.TRAINING_GLOBAL_STEP])
        training_global_step = run_state[self.TRAINING_GLOBAL_STEP]
        run_state[self.TRAINING_GLOBAL_STEP] = training_global_step + num_iterations
        self.update_run_state(run_id, run_state)

        return agent

    @staticmethod
    def _train(agent, replay_buffer, num_iterations, mini_batch_size):
        """
        Go through an update cycle on an Agent. The given agent's parameters are updated.

        :param agent: The Agent to update
        :param replay_buffer: A replay buffer to get trajectories from
        :param num_iterations: The number of epochs to run
        :param mini_batch_size: The number of trajectories to put in each mini-batch
        """

        start_time = time.time()

        logger.info("Starting training process RL Agent. time: %s" % str(start_time))

        # train the agent
        training_progress = None

        for i in range(num_iterations):

            replay_buffer.pre_process(i)

            traj, traj_meta = replay_buffer.get_batch(mini_batch_size)
            weights = traj_meta.probabilities if traj_meta else None
            loss_info = agent.train(experience=traj, weights=weights)

            replay_buffer.post_process(traj_meta, loss_info, i)

            curr_progress = int(i * 100 / num_iterations)
            if not training_progress or curr_progress > training_progress:
                training_progress = curr_progress
                logger.info("Current progress: %s Percent. Loss: %s" % (str(curr_progress), str(loss_info.loss)))

        end_time = time.time()
        total_time_second = int(end_time - start_time)
        logger.info("Agent training is complete. Total training time: %s minutes" % str(total_time_second / 60))
