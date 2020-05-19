from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rl_bakery.example.cartpole import ExampleCartPole
from unittest import TestCase
import yaml


class DQNTest(TestCase):
    def testDQNCartPole(self):
        with open('integration_test/hparams.yml', 'rb') as f:
            dqn_conf = yaml.load(f.read())['dqn']    # load the config file
        cartpole = ExampleCartPole(**dqn_conf)
        avg_rwd = cartpole.run()
        # evaluate over the last 3 runs
        eval_runs = 3
        self.assertGreater(sum(avg_rwd[-eval_runs:])/eval_runs, 185)