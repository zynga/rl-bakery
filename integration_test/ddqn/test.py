""" Test DDQN Algorithm """
import sys
from rl_bakery.example.cartpole_ddqn import make_runner


class DDQNTest():
    """ DDQN Test Class """
    def test_ddqn_cart_pole(self):
        """ Test Average Rewards of Cartpole using the DDQN agent (average over last 3 runs) """
        cartpole = make_runner()
        avg_rwd = cartpole.run()
        # evaluate over the last 3 runs
        eval_runs = 3
        assert(sum(avg_rwd[-eval_runs:])/eval_runs >= 185)


if __name__ == '__main__':
    print("Running Test for: %s" % __file__)
    DDQNTest().test_ddqn_cart_pole()
