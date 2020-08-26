import mlflow
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, rand
from hyperopt import SparkTrials
from functools import partial
import random
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers

import numpy as np

#
# A hacky way of getting hyperopt to work with GridSearch.
#
from hyperopt import pyll
from hyperopt.base import miscs_update_idxs_vals

_counter = 0 # Don't move this to a separate module.

class _GlobalCounter():
    def randint(self, low=-1, high=-1, size=-1):
        global _counter
        ret = _counter
        _counter += 1
        return [ret]

def _suggest(new_ids, domain, trials, seed):
    # We override the RandomState class to return a counter of num of runs.
    # This works because "_suggest" is called in the main process only.

    # rng = np.random.RandomState(seed)
    rng = _GlobalCounter() 
    rval = []
    for ii, new_id in enumerate(new_ids):
        # -- sample new specs, idxs, vals
        idxs, vals = pyll.rec_eval(
            domain.s_idxs_vals, memo={domain.s_new_ids: [new_id], domain.s_rng: rng}
        )
        new_result = domain.new_result()
        new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
        miscs_update_idxs_vals([new_misc], idxs, vals)
        rval.extend(trials.new_trial_docs([new_id], [None], [new_result], [new_misc]))
    return rval


class RunManager(object):

    def __init__(self, env_obj, trainer_class, env_config={}):
        """
        :env_obj: A PyEnvironment object that represents the Environment. 
                    TODO: tf_agent has a bug where PyEnvironment object cannot be pickled
        :trainer_class: A Trainer class for the training workflow. This will later be initialized with different
        parameters for different runs.
        :env_config: Configuration dictionary for PyEnvironment
        """
        # Workaround for tf_agent bug with pre-built gym environments
        if type(env_obj) == wrappers.TimeLimit:
            self._env_obj = env_obj._env.gym.unwrapped.spec.id
        else:
            self._env_obj = env_obj
        self._trainer_class = trainer_class
        self._env_config = env_config
        self._experiments = {}
    
    def create_experiment(self, experiment_name, hp_config, artifact_location=None):
        """Create an experiment in MLFLow. Currently, it does not support restarting from experiments saved in a different session.

        :experiment_name: Name of the mlflow experiment (should be unique across all other experiments). Need to make sure experiment with same name doesn't exist.
        :artifact_location: Where to store artifacts, if None then a default location is chosen.
        :hp_config: List of parameters for GridSearch.
        :return: The mlflow experiment ID.
        """
        # MLFlow doesn't support permanent deletion of experiments from program code.
        # It has to be done from command line.
        # experiment = mlflow.get_experiment_by_name(experiment_name)
        # if experiment is not None:
        #     mlflow.delete_experiment(experiment.experiment_id)
        exp_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        self._experiments[exp_id] = hp_config
        return exp_id
        # else:
        #     return experiment.experiment_id

    @staticmethod
    def _new_eval_fn(experiment_id, parent_run_id, env_obj, trainer_class, env_config):
        """Creating a new eval function for Hyperopt. Since the inputs and the returned function will be serializaed
           and transferred to Spark cluster, the user needs to make sure these classes don't load data from external
           files which the cluster may not have access to.
        
        :experiment_id: The experiment id where the eval function will run under.
        :parent_run_id: Parent run id of the eval function.
        :env_obj: The PyEnvironment for the run.
        :trainer_class: Trainer class for the run.
        :env_config: Environment configuration dictionary.
        """
        def eval_func(params):
            """The eval function for Hyperopts.
            :params: A dict containing chosen hyperparameters (by hyperopt) to be used for a single run
            :returns: A float which is used to compare different runs            
            """
            with mlflow.start_run(run_id=parent_run_id, experiment_id=experiment_id):
                with mlflow.start_run(experiment_id=experiment_id, nested=True) as child_run:
                    # TODO: There is a tf_agent bug with gym wrappers. Unpickling object from this class
                    # will cause infinite recursion loop. Here is a hacky way to reload the environment to avoid this.
                    # if type(env_obj) == wrappers.TimeLimit:
                    #     env_obj = suite_gym.load(env_obj._env.gym.unwrapped.spec.id)
                    # Workaround for tf agent bug
                    if type(env_obj) == str:
                        local_env_obj = suite_gym.load(env_obj)
                    trainer = trainer_class(local_env_obj, **params)
                    avg_reward = trainer.run()
                    avg_reward = 10
                    mlflow.log_params(params)
                    mlflow.log_metric('avg_reward', avg_reward) # TODO: Returning avg_reward by default for now.
            return avg_reward * -1 # Use negative reward for calculating fmin

        return eval_func

    def run(self, experiment_id, parallelism):
        """Run GridSearch with Hyperopt on Spark Cluster.
        :experiment_id: The experiment id to be run under.
        :parallelism: How many trials to run in parallel (must be > 0)

        :returns: The best hyper parameter combination and reward value.
        """
        # Reset global counter before each GridSearch.
        global _counter
        _counter = 0

        spark_trials = SparkTrials(parallelism=parallelism)
        
        hp_config = self._experiments[experiment_id]
        with mlflow.start_run(experiment_id=experiment_id) as parent_run:
            space = hp.choice("best_reward", hp_config)
            best_idx = fmin(fn=RunManager._new_eval_fn(experiment_id,
                                                       parent_run.info.run_id, 
                                                       self._env_obj,                                                   
                                                       self._trainer_class,
                                                       self._env_config),
                            space=space,
                            algo=_suggest,
                            trials=spark_trials,
                            max_evals=len(hp_config))   
            best_idx = int(best_idx["best_reward"])
            
            # Get results from SparkTrials
            best_reward = None
            for trail in spark_trials.trials:
                if int(trail['misc']['vals']['best_reward'][0]) == best_idx:
                    best_reward = trail['result']['loss']
                    break

            return {
                    "best_hp": hp_config[best_idx],
                    "best_reward": best_reward * -1
                }