from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rl_bakery.example.cartpole import ExampleCartPole
from unittest import TestCase


class TrainEval(TestCase):
    def testDQNCartPole(self):

        cartpole = ExampleCartPole(tb_path="/tmp/rl_application/dqn_cartpole_unit_test",
                                   learning_rate=1e-4, mini_batch_size=128, num_iterations=5000)
        avg_rwd = cartpole.run()
        eval_runs = 3
        self.assertGreater(sum(avg_rwd[-eval_runs:])/eval_runs, 185)
