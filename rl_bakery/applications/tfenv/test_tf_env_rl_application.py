from datetime import datetime, timedelta
from unittest import TestCase
from tf_agents.environments import tf_py_environment, suite_gym
from rl_bakery.applications.tfenv.indexed_tf_env import IndexedTFEnv
from rl_bakery.applications.tfenv.tf_env_rl_application import TFEnvRLApplication
from rl_bakery.spark_utilities import get_spark_session
from unittest.mock import patch


class TestRLEnvRLApplication(TestCase):
    @patch('rl_bakery.data_manager.data_manager.DataManager', autospec=True)
    def test_init_application(self, mock_dm):
        # init a rl env application
        envs = []
        for i in range(2):
            envs.append(IndexedTFEnv(tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0')), i))

        training_config = {
            "fc_layer_params": (100,),
            "learning_rate": 0.001,
            "agent_discount": 0.99,
            "mini_batch_size": 64,
            "num_training_iterations": 10000,
            "epsilon_greediness": 0.1,
            "gradient_clipping": 1.0
        }

        steps_num_per_run = 3

        spark_session = get_spark_session()
        app = TFEnvRLApplication(envs, spark_session, training_config, steps_num_per_run,
                                 engine_start_dt=datetime.now(), engine_training_interval=timedelta(days=1),
                                 num_partitions=2)
        self.assertListEqual(app.obs_cols, ['ob_0', 'ob_1', 'ob_2', 'ob_3'])
