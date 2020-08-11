""" Test DDQN Algorithm """
import sys
import yaml
from rl_bakery.example.yaml_parser import SimulatedEnviroment


class DDQNTest():
    """ DDQN Test Class """
    def test_ddqn_cart_pole(self):
        """ Test Average Rewards of Cartpole using the DDQN agent (average over last 3 runs) """
        cartpole = SimulatedEnviroment("cartpole_ddqn.yml")
        avg_rwd = cartpole.run()
        # evaluate over the last 3 runs
        eval_runs = 3
        assert(sum(avg_rwd[-eval_runs:])/eval_runs >= 185)


if __name__ == '__main__':
    print("Running Test for: %s" % __file__)
    DDQNTest().test_ddqn_cart_pole()
