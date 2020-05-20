""" Test DDQN Algorithm """
import sys
import yaml
sys.path.append('../rl-bakery/')
from rl_bakery.example.cartpole_ddqn import ExampleCartPoleDDQN


class DDQNTest():
    """ DDQN Test Class """
    def test_ddqn_cart_pole(self):
        """ Test Average Rewards of Cartpole using the DDQN agent (average over last 3 runs) """
        with open('integration_test/hparams.yml', 'rb') as config:
            ddqn_conf = yaml.load(config.read())['ddqn']    # load the config file
        cartpole = ExampleCartPoleDDQN(**ddqn_conf)
        avg_rwd = cartpole.run()
        # evaluate over the last 3 runs
        eval_runs = 3
        assert(sum(avg_rwd[-eval_runs:])/eval_runs >= 185)


if __name__ == '__main__':
    print("Running Test for: %s" % __file__)
    DDQNTest().test_ddqn_cart_pole()
