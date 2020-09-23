from rl_bakery.applications import agent_application
from rl_bakery.applications.simulation_runner import SimulationRunner, make_batch_tfenv
from rl_bakery.agent_trainer.config import QConfig
from rl_bakery.agent_trainer.agents import DDQNAgent
from rl_bakery.data_manager.builder import build_inmemory_data_manager
from rl_bakery.spark_utilities import get_spark_session

from tf_agents.environments import suite_gym, tf_py_environment

import logging
import time
from datetime import timedelta, datetime

from omegaconf import OmegaConf



def make_env():
    return tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))



def make_runner(num_runs=4, num_eval_episodes=100, eval_interval=1):
    params = ["agent.optimizer.learning_rate=0.01",
              "training.num_iterations=10000",
              "policy.eps_start=1.0",
              "policy.eps_final=0.1",
              "agent.fc_layer_params=[100,]",
              "trajectory.trajectory_training_window=100",
              "project.application_name=cartpole_sim",
              "project.dm_storage_path=/tmp/rl_applications/cartpole_sim/%s/" % int(time.time()),
              "project.tensorboard_path=/tmp/tb_path/cartpole_sim/%s" % datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
              ]
    conf = agent_application.make_config(QConfig(), params)
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
        agent_trainer=DDQNAgent(data_spec, conf),
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
