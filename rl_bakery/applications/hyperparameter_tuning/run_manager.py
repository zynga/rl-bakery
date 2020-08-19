import mlflow
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt import SparkTrials
from functools import partial

class RunManager(object):

    def __init__(self, simulator, agent, simulator_config, hyperparameter_config):
        """
            :simulator: should be a PyEnviroment/ OpenAI Gym Enviroment
            :agent: should be a wrapper around the tf_agent.TFAgent class
            :simulator_config: any parameters needed for the simulator
            :hyperparameter_config: a python dictionary of hyperparams representing a hyperopt search space
        """
        self._sim = simulator
        self._agent = agent
        self._sim_conf = simulator_config
        self._hyper_conf = hyperparameter_config
        self._experiment_list = set()
    
    def create_experiment(self, experiment_name, artifact_location=None):
        """
            :experiment_name: name of the mlflow experiment (should be unique across all other experiments)
            :artifact_location: where to store artifacts, if None then a default location is chose
            
            :returns: the mlflow experiment ID 
        """
        exp_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        self._experiment_list.add(exp_id)
        return exp_id
    
    def _eval_fn(self, params):
        """
            :params: a dict containing chosen hyperparameters (by hyperopt) to be used for a single run

            :returns: a float which is used to compare different runs
        """
        raise Exception('Implement Eval Method')

    def run(self, experiment_id, max_evals, parallelism, algo=tpe.suggest):
        """
            :experiment_id: The experiment id to be run (should be a result from self.create_experiment)
            :max_evals: the number of hyperopt trials to run (must be > 0)
            :parallelism: how many trials to run in parallel (must be > 0)
            :algo: Algorithm used by hyperOpt to search in the hyper param space

            :returns: the best hyperparameter combination
        """

        spark_trials = SparkTrials(parallelism=parallelism)
        
        with mlflow.start_run(experiment_id=experiment_id):
            return fmin(fn=self._eval_fn,
                        space=self._hyper_conf,
                        return_argmin=True,
                        trials=spark_trials,
                        algo=algo,
                        max_evals=max_evals)