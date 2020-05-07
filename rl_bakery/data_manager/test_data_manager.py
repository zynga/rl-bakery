from moto import mock_s3
from rl_bakery.applications.tf_rl_application import MockRLApplication
from rl_bakery.data_manager.data_manager import MissingDataException, ParquetSparkStorage, TFAgentStorage, S3JSONStorage
from rl_bakery.spark_utilities import PySparkTestCase, compare_dataframes
from unittest import TestCase

import shutil


class DataManagerTest(PySparkTestCase, TestCase):
    def test_get_tf_agent_success(self):
        mock_rl_app = MockRLApplication()
        test_agent = mock_rl_app.init_agent()
        root_path = "test_parquet_path_root"
        storage = TFAgentStorage(mock_rl_app, root_path)
        run_id = 3
        storage.store(test_agent, run_id)

        returned_agent = storage.get(run_id)
        self.assertEquals(test_agent, returned_agent)

    def test_get_missing_tf_agent(self):
        mock_rl_app = MockRLApplication()
        root_path = "test_parquet_path_root"
        storage = TFAgentStorage(mock_rl_app, root_path)

        run_id = 2
        with self.assertRaises(MissingDataException):
            storage.get(run_id)

    def test_get_parquet_spark_success(self):
        test_df = self.spark.createDataFrame([{"player_id": 123, "game_id": 345, "col1": 100},
                                              {"player_id": 124, "game_id": 345, "col1": 200},
                                              {"player_id": 125, "game_id": 345, "col1": 300}])
        root_path = "test_parquet_path_root"
        storage = ParquetSparkStorage(root_path)
        storage.store(test_df, 1)

        return_df = storage.get(1)
        compare_dataframes(test_df, return_df)

        # remove data created by test
        shutil.rmtree(root_path, ignore_errors=False, onerror=None)

    def test_get_parquet_spark_missing_data(self):
        test_df = self.spark.createDataFrame([{"player_id": 123, "game_id": 345, "col1": 100},
                                              {"player_id": 124, "game_id": 345, "col1": 200},
                                              {"player_id": 125, "game_id": 345, "col1": 300}])
        root_path = "test_parquet_path_root"
        storage = ParquetSparkStorage(root_path)
        storage.store(test_df, 1)

        with self.assertRaises(MissingDataException):
            storage.get(2)

        # remove data created by test
        shutil.rmtree(root_path, ignore_errors=False, onerror=None)

    @mock_s3
    def test_s3_json_storage_success(self):
        bucket = "mock_bucket"
        root_path = "s3://{}/mock_key/".format(bucket)
        storage = S3JSONStorage(root_path)
        storage.s3_client.create_bucket(Bucket=bucket)
        data = {
            "test_date": 5,
            "tuple_data": [("abc", 12)]
        }
        storage.store(data, 1)

        result = storage.get(1)
        self.assertDictEqual(data, result)

    @mock_s3
    def test_s3_json_storage_missing_data(self):
        bucket = "mock_bucket"
        root_path = "s3://{}/mock_key/".format(bucket)
        storage = S3JSONStorage(root_path)
        storage.s3_client.create_bucket(Bucket=bucket)
        data = {
            "test_date": 5,
            "tuple_data": [("abc", 12)]
        }

        storage.store(data, 1)
        with self.assertRaises(MissingDataException):
            storage.get(2)

    def test_get_bucket_and_key_success(self):
        test_s3_path = "s3://fake_bucket/fake_key/1/2"
        bucket, key = S3JSONStorage._get_bucket_and_key(test_s3_path)
        self.assertEqual(bucket, "fake_bucket")
        self.assertEqual(key, "fake_key/1/2")

        test_s3_path = "s3a://fake_bucket/fake_key/1/2"
        bucket, key = S3JSONStorage._get_bucket_and_key(test_s3_path)
        self.assertEqual(bucket, "fake_bucket")
        self.assertEqual(key, "fake_key/1/2")
