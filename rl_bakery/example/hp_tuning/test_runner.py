from hyperopt import hp
from rl_bakery.engine.run_manager import RunManager
from tf_agents.environments import suite_gym
from rl_bakery.engine.dqn_trainer import DQNTrainer
from config import hp_config

if __name__ == '__main__':
    env_obj = suite_gym.load('CartPole-v0')
    run_mgr = RunManager(env_obj, DQNTrainer)
    exp_id = run_mgr.create_experiment('test_exp_1', hp_config=hp_config)
    best = run_mgr.run(exp_id, parallelism=2)
    print(best)