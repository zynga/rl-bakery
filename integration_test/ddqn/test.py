""" Test DDQN Algorithm """
from unittest import TestCase
import yaml
from rl_bakery.example.cartpole_ddqn import ExampleCartPoleDDQN


class DDQNTest(TestCase):
    """ DDQN Test Class """
    def test_ddqn_cart_pole(self):
        """ Test Average Rewards of Cartpole using the DDQN agent (average over last 3 runs) """
        with open('integration_test/hparams.yml', 'rb') as config:
            ddqn_conf = yaml.load(config.read())['ddqn']    # load the config file
        cartpole = ExampleCartPoleDDQN(**ddqn_conf)
        avg_rwd = cartpole.run()
        # evaluate over the last 3 runs
        eval_runs = 3
        self.assertGreater(sum(avg_rwd[-eval_runs:])/eval_runs, 185)
