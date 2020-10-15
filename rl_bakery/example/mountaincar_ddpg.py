""" MountainCarContinuous-V0 with a ddpg agent """

from rl_bakery.applications import agent_application
from rl_bakery.applications.simulation_runner import SimulationRunner, make_batch_tfenv
from rl_bakery.agent_trainer.ddpg import DDPGAgent, DDPGConfig
from rl_bakery.data_manager.builder import build_inmemory_data_manager
from rl_bakery.spark_utilities import get_spark_session

from tf_agents.environments import suite_gym, tf_py_environment

import logging
import time
from datetime import timedelta, datetime

from omegaconf import OmegaConf


def make_env():
    # function to create a tf environment
    return tf_py_environment.TFPyEnvironment(suite_gym.load("MountainCarContinuous-v0"))


def make_runner(num_runs = 4, num_eval_episodes = 100, eval_interval = 1):
    params = ["agent.actor_optimizer.learning_rate=1e-4",
              "agent.critic_optimizer.learning_rate=1e-3",
              "training.num_iterations=2000",
              "env.num_envs=60",
              "env.num_steps_per_run=50",
              "policy.eps_start=1.0",
              "policy.eps_final=0.1",
              "policy.eps_steps=1000",
              "agent.actor_fc_layer_params=[400, 300]",
              "agent.observation_fc_layer_params=[400,]",
              "agent.joint_fc_layer_params=[300,]",
              "trajectory.trajectory_training_window=100",
              "project.application_name=mountaincar_ddpg",
              "project.dm_storage_path=/tmp/rl_applications/mountaincar_sim/%s/" % int(time.time()),
              "project.tensorboard_path=/tmp/tb_path/mountaincar_sim/%s" % datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
              ]
    conf = agent_application.make_config(DDPGConfig(), params)
    print(OmegaConf.to_yaml(conf))

    first_timestep_dt = datetime(year=2019, month=8, day=7, hour=10)
    training_interval = timedelta(days=1)
    spark = get_spark_session()
    tfenv = make_batch_tfenv(make_env, conf, first_timestep_dt, training_interval, spark)

    data_spec = agent_application.DataSpec(
        action_spec=tfenv.action_spec,
        observation_spec=tfenv.observation_spec
    )

    application = agent_application.AgentApplication(
        data_spec=data_spec,
        agent_trainer=DDPGAgent(data_spec, conf),
        env=tfenv,
        config=conf,
        first_timestep_dt=first_timestep_dt,
        training_interval=training_interval
    )

    dm = build_inmemory_data_manager(application)
    tfenv.set_dm(dm)
    return SimulationRunner(application=application, make_eval_env=make_env, dm=dm,
                            num_runs=num_runs, num_eval_episodes=num_eval_episodes, eval_interval=eval_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runner = make_runner()
    runner.run()
