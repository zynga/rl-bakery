""" Test DQN Algorithm """
from unittest import TestCase
import yaml
from rl_bakery.example.cartpole import ExampleCartPole


class DQNTest(TestCase):
    """ DDQN Test Class """
    def test_dqn_cart_pole(self):
        """ Test Average Rewards of Cartpole using the DQN agent (average over last 3 runs) """
        with open('integration_test/hparams.yml', 'rb') as config:
            dqn_conf = yaml.load(config.read())['dqn']    # load the config file
        cartpole = ExampleCartPole(**dqn_conf)
        avg_rwd = cartpole.run()
        # evaluate over the last 3 runs
        eval_runs = 3
        self.assertGreater(sum(avg_rwd[-eval_runs:])/eval_runs, 185)
