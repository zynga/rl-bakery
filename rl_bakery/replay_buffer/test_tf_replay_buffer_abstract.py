from rl_bakery.replay_buffer.tf_replay_buffer_abstract import convert_data_to_tensor
from rl_bakery.spark_utilities import PySparkTestCase
from tf_agents.utils import test_utils
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape


class TestConvertDataToTensor(PySparkTestCase, test_utils.TestCase):
    def test_traj_converter_tensor_with_casting_float(self):

        data = [1]
        data_spec = tf.constant(0, dtype=tf.float32)

        data_tensor, spec = convert_data_to_tensor(data, data_spec)

        expected_tensor = tf.constant([1.0], dtype=tf.float32)
        expected_spec = tf.TensorSpec(TensorShape([]), tf.float32)

        self.assertAllEqual(data_tensor, expected_tensor)
        self.assertAllEqual(spec, expected_spec)

    def test_traj_converter_tensor_with_casting_int(self):

        data = [1]
        data_spec = tf.constant(0, dtype=tf.int32)

        data_tensor, spec = convert_data_to_tensor(data, data_spec)

        expected_tensor = tf.constant([1], dtype=tf.int32)
        expected_spec = tf.TensorSpec(TensorShape([]), tf.int32)
        bad_spec = tf.TensorSpec(TensorShape([]), tf.float32)

        self.assertAllEqual(data_tensor, expected_tensor)
        self.assertAllEqual(spec, expected_spec)
        self.assertNotEqual(spec, bad_spec)

    def test_traj_converter_dict_to_tensor(self):

        data = {"k": [1]}
        data_spec = {"k": tf.constant(0, dtype=tf.float32)}

        data_tensor, spec = convert_data_to_tensor(data, data_spec)

        expected_tensor = {"k": tf.constant([1.0], dtype=tf.float32)}
        expected_spec = {"k": tf.TensorSpec(TensorShape([]))}

        self.assertAllEqual(data_tensor, expected_tensor)
        self.assertAllEqual(spec, expected_spec)

    def test_traj_converter_empty_list(self):

        data = ()
        data_spec = ()

        data_tensor, spec = convert_data_to_tensor(data, data_spec)

        expected_tensor = ()
        expected_spec = ()

        self.assertAllEqual(data_tensor, expected_tensor)
        self.assertAllEqual(spec, expected_spec)
