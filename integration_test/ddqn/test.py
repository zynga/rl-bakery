from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rl_bakery.example.cartpole_ddqn import ExampleCartPoleDDQN
from unittest import TestCase
import yaml


class DDQNTest(TestCase):
    def testDQNCartPole(self):
        with open('integration_test/hparams.yml', 'rb') as f:
            ddqn_conf = yaml.load(f.read())['ddqn']    # load the config file
        cartpole = ExampleCartPoleDDQN(**ddqn_conf)
        avg_rwd = cartpole.run()
        # evaluate over the last 3 runs
        eval_runs = 3
        self.assertGreater(sum(avg_rwd[-eval_runs:])/eval_runs, 185)