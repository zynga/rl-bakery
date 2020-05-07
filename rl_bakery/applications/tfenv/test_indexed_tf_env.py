from unittest import TestCase
from rl_bakery.applications.tfenv.indexed_tf_env import IndexedTFEnv
from tf_agents.environments import tf_py_environment, suite_gym
from tf_agents.trajectories.policy_step import PolicyStep
import tensorflow as tf


class TestRLGymEnv(TestCase):
    def test_step(self):
        tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))
        indexed_tf_env = IndexedTFEnv(tf_env, 5)
        # take first action
        a1 = PolicyStep(action=tf.convert_to_tensor([1]), state=(), info=())
        time_step_0 = indexed_tf_env.step(a1)
        self.assertEqual(time_step_0["env_id"], 5)
        self.assertEqual(time_step_0["ts_id"], 0)
        self.assertEqual(time_step_0["reward"], 0)
        self.assertEqual(time_step_0["step_type"], 0)
        self.assertEqual(time_step_0["discount"], 1.0)
        self.assertTrue("ob_0" in time_step_0)
        self.assertTrue("ob_1" in time_step_0)
        self.assertTrue("ob_2" in time_step_0)
        self.assertTrue("ob_3" in time_step_0)

        # take second action
        a2 = PolicyStep(action=tf.convert_to_tensor([0]), state=(), info=())
        time_step_1 = indexed_tf_env.step(a2)
        self.assertEqual(time_step_1["env_id"], 5)
        self.assertEqual(time_step_1["ts_id"], 1)
        self.assertEqual(time_step_1["reward"], 1)
        self.assertEqual(time_step_1["step_type"], 1)
        self.assertEqual(time_step_1["discount"], 1.0)
        self.assertTrue("ob_0" in time_step_1)
        self.assertTrue("ob_1" in time_step_1)
        self.assertTrue("ob_2" in time_step_1)
        self.assertTrue("ob_3" in time_step_1)
