from rl_bakery.applications.tf_rl_application import TFRLApplication
from rl_bakery.data_manager.data_manager import DATANAME
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
import tensorflow as tf


class TFEnvRLApplication(TFRLApplication):
    # TODO This legacy class should be refactored to read all configuration values from an ApplicationConfig and follow
    # a more generic pattern for retrieving trajectories

    def __init__(self, envs, spark_session, training_config, steps_num_per_run, engine_start_dt,
                 engine_training_interval, num_partitions=10):
        """
        This adapter allows which allows the batching of multiple simulators in a single application in addition to
        control the number of interaction between the agent and the environment within 1 training run.

        Params:
            envs (list): list of gym simulated environments
            spark_session: A Spark session, which is used to build and join timesteps
            training_config: A dictionary
            steps_num_per_run (int): number of interactions between the agent and the simulator within 1 training run
            agent_config: TBD
            engine_start_dt (datetime): The engine start datetime. This is needed by buildtimestep() to compute
                             the current_run_id
            training_interval (timedelta): timedelta between 2 runs. This is needed by buildtimestep() to compute the
                                           current_run_id
            num_partitions (int): number of spark partitions when build the timestep dataframe
        """

        if not len(envs):
            raise ValueError("Envs must be list of at least one TFAgent environments.")

        self._envs = envs
        self._spark = spark_session
        self._steps_num_per_run = steps_num_per_run
        self._num_partitions = num_partitions

        self._engine_start_dt = engine_start_dt
        self._engine_training_interval = engine_training_interval

        self._training_config = training_config
        self._dm = None

        # validate spec for each env
        if any(env.tf_env.action_spec() != envs[0].tf_env.action_spec() for env in envs):
            raise ValueError(
                "All environments must have the same action spec.  Saw: %s" %
                [env.action_spec() for env in envs])
        if any(env.tf_env.observation_spec() != envs[0].tf_env.observation_spec() for env in envs):
            raise ValueError(
                "All environments must have the same observation_spec.  Saw: %s" %
                [env.observation_spec() for env in envs])

        self._step_counter = 0
        self._num_episodes = 0

    @property
    def env_id_cols(self):
        return ["env_id"]

    @property
    def ts_id_col(self):
        return "ts_id"

    @property
    def obs_cols(self):
        spec_shape = self._envs[0].tf_env.observation_spec().shape
        return ["ob_{}".format(i) for i in range(spec_shape[0])]

    @property
    def observation_spec(self):
        return tensor_spec.from_spec(self._envs[0].tf_env.observation_spec())

    @property
    def action_spec(self):
        return tensor_spec.from_spec(self._envs[0].tf_env.action_spec())

    @property
    def training_config(self):
        return self._training_config

    def set_dm(self, dm):
        self._dm = dm

    def build_time_steps(self, previous_run_dt, current_run_dt):
        # compute the previous run_id
        previous_run_id = int((previous_run_dt - self._engine_start_dt) / self._engine_training_interval)

        # get policy trained in the previous run
        agent_policy = self._get_policy(previous_run_id)

        num_collect_steps = self._steps_num_per_run
        if previous_run_id == 0:
            num_collect_steps = int(self._training_config["initial_collect_steps"] / len(self._envs))

        time_steps_list = []
        for rl_env in self._envs:
            for _ in range(num_collect_steps):
                tf_env_time_step = rl_env.tf_env.current_time_step()
                action_step = agent_policy.action(tf_env_time_step)
                rl_time_step = rl_env.step(action_step)
                time_steps_list.append(rl_time_step)
                self._step_counter += 1
                if int(rl_time_step["step_type"]) == int(ts.StepType.LAST):
                    self._num_episodes += 1

        tf.compat.v2.summary.scalar(name="env_metrics/EnvironmentSteps", data=self._step_counter)
        tf.compat.v2.summary.scalar(name="env_metrics/EnvironmentEpisodes", data=self._num_episodes)
        tf.compat.v2.summary.scalar(name="env_metrics/EnvironmentNumber", data=len(self._envs))

        time_step_df = self._spark.createDataFrame(time_steps_list).coalesce(self._num_partitions)
        return time_step_df

    def _get_policy(self, run_id):

        agent = self._dm.get(DATANAME.MODEL, run_id)
        agent_policy = agent.collect_policy

        if isinstance(agent_policy, EpsilonGreedyPolicy):
            eps_start = self._training_config.get("eps_start", 1.0)
            eps_final = self._training_config.get("eps_final", 0.1)
            eps_steps = self._training_config.get("eps_steps", 1)

            print("eps_start: %s eps_final: %s eps_steps: %s" % (eps_start, eps_final, eps_steps))
            if eps_steps > self._step_counter:
                epsilon = eps_final + (eps_start - eps_final) * (eps_steps - self._step_counter) / eps_steps
            else:
                epsilon = eps_final

            # TODO avoid writing to private attribute
            agent_policy._epsilon = epsilon

            print("Collection Epsilon: %s" % str(epsilon))
            tf.compat.v2.summary.scalar(name="env_metrics/exploration_rate", data=epsilon)

        return agent_policy
