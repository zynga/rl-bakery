""" Test DDPG Algorithm """
from rl_bakery.example.mountaincar_ddpg import make_runner


class DDPGTest():
    """ DDPG Test Class """
    def test_ddpg_mountain_car(self):
        """ Test Average Rewards of MountainCarContinuous using the DDPG agent (average over last 3 runs) """
        mountaincar = make_runner()
        avg_rwd = mountaincar.run()
        # evaluate over the last 3 runs
        eval_runs = 3
        assert(sum(avg_rwd[-eval_runs:])/eval_runs >= 80)


if __name__ == '__main__':
    print("Running Test for: %s" % __file__)
    DDPGTest().test_ddpg_mountain_car()
