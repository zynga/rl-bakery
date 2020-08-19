from hyperopt import hp
from rl_bakery.applications.hyperparameter_tuning.run_manager import RunManager

class BasicRunner(RunManager):

    def __init__(self):
        space = {
            'a': hp.choice('a',
            [
                ('case 1', 1 + hp.lognormal('c1', 0, 1)),
                ('case 2', hp.uniform('c2', -10, 10))
            ])
        }
        super(BasicRunner, self).__init__(None, None, {}, space)

    def _eval_fn(self, params):
        case, val = params['a']
        if case == 'case 1':
            return val
        else:
            return val ** 2

if __name__ == '__main__':
    basic_runner = BasicRunner()
    exp_id = basic_runner.create_experiment('test_exp_1')
    best = basic_runner.run(exp_id, 1, 1)
    print(best)