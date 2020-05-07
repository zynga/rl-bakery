from rl_bakery.operation.trajectory_builder import TrajectoryBuilder
from rl_bakery.spark_utilities import PySparkTestCase
from unittest import TestCase
from datetime import datetime


class TrajectoryBuilderTest(PySparkTestCase, TestCase):
    maxDiff = None

    def test_timestep_operation(self):
        """
        * 1 user with 3 timesteps
        * 1 user with 2 timesteps
        * 1 user with 1 timesteps
        """

        mock_timestep = [
            {"env_id": 1, "ts_id": 0, "obs": 0, "action": 0, "reward": 0.0, "step_type": 0, "discount": 0.0},
            {"env_id": 1, "ts_id": 1, "obs": 1, "action": 1, "reward": 0.1, "step_type": 1, "discount": 0.1},
            {"env_id": 1, "ts_id": 2, "obs": 2, "action": 2, "reward": 0.2, "step_type": 2, "discount": 0.2},
            {"env_id": 2, "ts_id": 0, "obs": 3, "action": 3, "reward": 0.3, "step_type": 0, "discount": 0.3},
            {"env_id": 2, "ts_id": 1, "obs": 4, "action": 4, "reward": 0.4, "step_type": 1, "discount": 0.4},
            {"env_id": 3, "ts_id": 0, "obs": 5, "action": 5, "reward": 0.5, "step_type": 0, "discount": 0.5}
        ]

        env_id_cols = ["env_id"]
        ts_id_col = "ts_id"
        obs_cols = ["obs"]
        n_step = 1
        tb = TrajectoryBuilder(obs_cols=obs_cols, env_id_cols=env_id_cols, ts_id_col=ts_id_col, n_step=n_step)
        res = tb.run(mock_timestep)

        expected_result = {
            "observation": [[[0], [1]], [[1], [2]], [[3], [4]]],
            "step_type": [[0, 1], [1, 2], [0, 1]],
            "action": [[0, 1], [1, 2], [3, 4]],
            "reward": [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]],
            "discount": [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]],
            "next_step_type": [[1, 1], [2, 2], [1, 1]],
            "policy_info": ()
        }

        self.assertEquals(res, expected_result)

    def test_non_sufficient_timestep_operation(self):
        """
        * all user with 1 timestep
        """

        mock_timestep = [
            {"env_id": 3, "ts_id": 0, "obs": 5, "action": 5, "reward": 0.5, "step_type": 0, "discount": 0.5},
            {"env_id": 1, "ts_id": 1, "obs": 1, "action": 1, "reward": 0.1, "step_type": 1, "discount": 0.1},
            {"env_id": 2, "ts_id": 1, "obs": 4, "action": 4, "reward": 0.4, "step_type": 1, "discount": 0.4},

        ]

        env_id_cols = ["env_id"]
        ts_id_col = "ts_id"
        obs_cols = ["obs"]
        n_step = 1
        tb = TrajectoryBuilder(obs_cols=obs_cols, env_id_cols=env_id_cols, ts_id_col=ts_id_col, n_step=n_step)
        res = tb.run(mock_timestep)
        expected_result = None

        self.assertEquals(res, expected_result)

    def test_terminated_timestep_operation(self):
        """
        1 user with 5 timesteps but with the 3rd being terminal
        """

        mock_timestep = [
            {"env_id": 1, "ts_id": 0, "obs": 0, "action": 0, "reward": 0.0, "step_type": 0, "discount": 0.0},
            {"env_id": 1, "ts_id": 1, "obs": 1, "action": 1, "reward": 0.1, "step_type": 1, "discount": 0.1},
            {"env_id": 1, "ts_id": 2, "obs": 2, "action": 2, "reward": 0.2, "step_type": 2, "discount": 0.2},
            {"env_id": 1, "ts_id": 3, "obs": 3, "action": 3, "reward": 0.3, "step_type": 0, "discount": 0.3},
            {"env_id": 1, "ts_id": 4, "obs": 4, "action": 4, "reward": 0.4, "step_type": 1, "discount": 0.4}
        ]

        env_id_cols = ["env_id"]
        ts_id_col = "ts_id"
        obs_cols = ["obs"]
        n_step = 1
        tb = TrajectoryBuilder(obs_cols=obs_cols, env_id_cols=env_id_cols, ts_id_col=ts_id_col, n_step=n_step)
        res = tb.run(mock_timestep)

        expected_result = {
            "observation": [[[0], [1]], [[1], [2]], [[3], [4]]],
            "step_type": [[0, 1], [1, 2], [0, 1]],
            "action": [[0, 1], [1, 2], [3, 4]],
            "reward": [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]],
            "discount": [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]],
            "next_step_type": [[1, 1], [2, 2], [1, 1]],
            "policy_info": ()
        }

        self.assertEquals(res, expected_result)

    def test_n_step_operation(self):
        """
        test using n_step larger than 1
        * 1 user with 3 timesteps
        * 1 user with 2 timesteps
        * 1 user with 1 timesteps
        """

        mock_timestep = [
            {"env_id": 1, "ts_id": 0, "obs": 0, "action": 0, "reward": 0.0, "step_type": 0, "discount": 0.0},
            {"env_id": 1, "ts_id": 1, "obs": 1, "action": 1, "reward": 0.1, "step_type": 1, "discount": 0.1},
            {"env_id": 1, "ts_id": 2, "obs": 2, "action": 2, "reward": 0.2, "step_type": 2, "discount": 0.2},
            {"env_id": 2, "ts_id": 0, "obs": 3, "action": 3, "reward": 0.3, "step_type": 0, "discount": 0.3},
            {"env_id": 2, "ts_id": 1, "obs": 4, "action": 4, "reward": 0.4, "step_type": 1, "discount": 0.4},
            {"env_id": 3, "ts_id": 0, "obs": 5, "action": 5, "reward": 0.5, "step_type": 0, "discount": 0.5}
        ]

        env_id_cols = ["env_id"]
        ts_id_col = "ts_id"
        obs_cols = ["obs"]
        n_step = 2
        tb = TrajectoryBuilder(obs_cols=obs_cols, env_id_cols=env_id_cols, ts_id_col=ts_id_col, n_step=n_step)
        res = tb.run(mock_timestep)

        expected_result = {
            "observation": [[[0], [1], [2]]],
            "step_type": [[0, 1, 2]],
            "action": [[0, 1, 2]],
            "reward": [[0.1, 0.2, 0.2]],
            "discount": [[0.1, 0.2, 0.2]],
            "next_step_type": [[1, 2, 2]],
            "policy_info": ()
        }

        self.assertEquals(res, expected_result)

    def test_timestep_multiple_cols_operation(self):
        """
        test timesteps that use multiple id and observation columns
        * 1 user with 3 timesteps
        * 1 user with 2 timesteps
        * 1 user with 1 timesteps
        """

        mock_timestep = [
            {"env_id_1": 1, "env_id_2": 1, "ts_id": 0, "obs_1": 0, "obs_2": 5, "action": 0, "reward": 0.0,
             "step_type": 0, "discount": 0.0},
            {"env_id_1": 1, "env_id_2": 1, "ts_id": 1, "obs_1": 1, "obs_2": 5, "action": 1, "reward": 0.1,
             "step_type": 1, "discount": 0.1},
            {"env_id_1": 1, "env_id_2": 1, "ts_id": 2, "obs_1": 2, "obs_2": 5, "action": 2, "reward": 0.2,
             "step_type": 2, "discount": 0.2},
            {"env_id_1": 2, "env_id_2": 1, "ts_id": 0, "obs_1": 3, "obs_2": 5, "action": 3, "reward": 0.3,
             "step_type": 0, "discount": 0.3},
            {"env_id_1": 2, "env_id_2": 1, "ts_id": 1, "obs_1": 4, "obs_2": 5, "action": 4, "reward": 0.4,
             "step_type": 1, "discount": 0.4},
            {"env_id_1": 3, "env_id_2": 1, "ts_id": 0, "obs_1": 5, "obs_2": 5, "action": 5, "reward": 0.5,
             "step_type": 0, "discount": 0.5}
        ]

        env_id_cols = ["env_id_1", "env_id_2"]
        ts_id_col = "ts_id"
        obs_cols = ["obs_1", "obs_2"]
        n_step = 1
        tb = TrajectoryBuilder(obs_cols=obs_cols, env_id_cols=env_id_cols, ts_id_col=ts_id_col, n_step=n_step)
        res = tb.run(mock_timestep)

        expected_result = {
            "observation": [[[0, 5], [1, 5]], [[1, 5], [2, 5]], [[3, 5], [4, 5]]],
            "step_type": [[0, 1], [1, 2], [0, 1]],
            "action": [[0, 1], [1, 2], [3, 4]],
            "reward": [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]],
            "discount": [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]],
            "next_step_type": [[1, 1], [2, 2], [1, 1]],
            "policy_info": ()
        }

        self.assertEquals(res, expected_result)

    def test_ts_id_datetime_operation(self):
        """
        * 1 user with 3 timesteps
        * 1 user with 2 timesteps
        * 1 user with 1 timesteps
        """

        mock_timestep = [
            {"env_id": 1, "ts_id": datetime(1970, 1, 1), "obs": 0, "action": 0, "reward": 0.0, "step_type": 0,
             "discount": 0.0},
            {"env_id": 1, "ts_id": datetime(1970, 1, 2), "obs": 1, "action": 1, "reward": 0.1, "step_type": 1,
             "discount": 0.1},
            {"env_id": 1, "ts_id": datetime(1970, 1, 3), "obs": 2, "action": 2, "reward": 0.2, "step_type": 2,
             "discount": 0.2},
            {"env_id": 2, "ts_id": datetime(1980, 1, 1), "obs": 3, "action": 3, "reward": 0.3, "step_type": 0,
             "discount": 0.3},
            {"env_id": 2, "ts_id": datetime(1980, 2, 1), "obs": 4, "action": 4, "reward": 0.4, "step_type": 1,
             "discount": 0.4},
            {"env_id": 3, "ts_id": datetime(1970, 1, 1), "obs": 5, "action": 5, "reward": 0.5, "step_type": 0,
             "discount": 0.5}
        ]

        env_id_cols = ["env_id"]
        ts_id_col = "ts_id"
        obs_cols = ["obs"]
        n_step = 1
        tb = TrajectoryBuilder(obs_cols=obs_cols, env_id_cols=env_id_cols, ts_id_col=ts_id_col, n_step=n_step)
        res = tb.run(mock_timestep)

        expected_result = {
            "observation": [[[0], [1]], [[1], [2]], [[3], [4]]],
            "step_type": [[0, 1], [1, 2], [0, 1]],
            "action": [[0, 1], [1, 2], [3, 4]],
            "reward": [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]],
            "discount": [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]],
            "next_step_type": [[1, 1], [2, 2], [1, 1]],
            "policy_info": ()
        }

        self.assertEquals(res, expected_result)

    def test_timestep_operation_with_policy_info(self):
        """
        * 1 user with 3 timesteps
        * 1 user with 2 timesteps
        * 1 user with 1 timesteps
        """
        meta_value_1 = [0.123, 0.876]
        meta_value_2 = 0.4
        meta_value_3 = [0.3, 0.7]
        meta_value_4 = 0.6
        policy_info_dict_1 = {"meta_1": meta_value_1, "meta_2": meta_value_2}
        policy_info_dict_2 = {"meta_1": meta_value_3, "meta_2": meta_value_4}
        mock_timestep = [
            {"env_id": 1, "ts_id": 0, "obs": 0, "action": 0, "reward": 0.0, "step_type": 0, "discount": 0.0,
             "policy_info": policy_info_dict_1},
            {"env_id": 1, "ts_id": 1, "obs": 1, "action": 1, "reward": 0.1, "step_type": 1, "discount": 0.1,
             "policy_info": policy_info_dict_2},
            {"env_id": 1, "ts_id": 2, "obs": 2, "action": 2, "reward": 0.2, "step_type": 2, "discount": 0.2,
             "policy_info": policy_info_dict_1},
            {"env_id": 2, "ts_id": 0, "obs": 3, "action": 3, "reward": 0.3, "step_type": 0, "discount": 0.3,
             "policy_info": policy_info_dict_2},
            {"env_id": 2, "ts_id": 1, "obs": 4, "action": 4, "reward": 0.4, "step_type": 1, "discount": 0.4,
             "policy_info": policy_info_dict_1},
            {"env_id": 3, "ts_id": 0, "obs": 5, "action": 5, "reward": 0.5, "step_type": 0, "discount": 0.5,
             "policy_info": policy_info_dict_1}
        ]

        env_id_cols = ["env_id"]
        ts_id_col = "ts_id"
        obs_cols = ["obs"]
        n_step = 1
        tb = TrajectoryBuilder(obs_cols=obs_cols, env_id_cols=env_id_cols, ts_id_col=ts_id_col, n_step=n_step)
        res = tb.run(mock_timestep)

        expected_result = {
            "observation": [[[0], [1]], [[1], [2]], [[3], [4]]],
            "step_type": [[0, 1], [1, 2], [0, 1]],
            "action": [[0, 1], [1, 2], [3, 4]],
            "reward": [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]],
            "discount": [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]],
            "next_step_type": [[1, 1], [2, 2], [1, 1]],
            "policy_info": {
                "meta_1": [[meta_value_1, meta_value_3], [meta_value_3, meta_value_1], [meta_value_3, meta_value_1]],
                "meta_2": [[meta_value_2, meta_value_4], [meta_value_4, meta_value_2], [meta_value_4, meta_value_2]]
            }
        }

        self.assertEquals(res, expected_result)
