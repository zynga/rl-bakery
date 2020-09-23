""" Test DQN Algorithm """
from rl_bakery.example.cartpole_dqn import make_runner

class DQNTest():
    """ DDQN Test Class """
    def test_dqn_cart_pole(self):
        """ Test Average Rewards of Cartpole using the DQN agent (average over last 3 runs) """
        cartpole = make_runner(num_runs=6)
        eval_rewards = cartpole.run()
        # evaluate over the last 3 runs
        eval_runs = 3
        print("Received evaluation rewards: %s" % eval_rewards)
        assert(sum(eval_rewards[-eval_runs:])/eval_runs >= 185)


if __name__ == '__main__':
    print("Running Test for: %s" % __file__)
    DQNTest().test_dqn_cart_pole()
