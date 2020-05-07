from botocore.exceptions import ClientError
from rl_bakery.spark_utilities import get_spark_session
from tf_agents.agents.tf_agent import TFAgent

import abc
import boto3
import logging
import os
import re
import tensorflow as tf

logger = logging.getLogger(__name__)


class MissingDataException(Exception):
    def __init__(self, message):
        self._message = message

    def __str__(self):
        return self._message


class DATANAME:
    TIMESTEP = "timestep"
    MODEL = "model"
    RUN_CONTEXT = "run_context"


class DataManager:
    def __init__(self):
        # A map of datanames to a DataStorage
        self.data_storages = {}

    def add_data(self, dataname, data_storage):
        self.data_storages[dataname] = data_storage
        return self

    def get(self, dataname, run_id):
        logger.debug("Getting %s for ts %s" % (dataname, run_id))

        ds = self.data_storages.get(dataname)
        if not ds:
            raise Exception("data %s is not known to DataManagerSet" % dataname)

        return ds.get(run_id)

    def store(self, data, dataname, run_id):
        logger.debug("Storing %s for ts %s" % (dataname, run_id))
        ds = self.data_storages.get(dataname)
        if not ds:
            raise Exception("data %s is not known to DataManagerSet" % dataname)

        return ds.store(data, run_id)

    def get_latest(self, data_name, run_id):
        """
        load latest data stored up to the provided run_id

        Params:
            data_name (DATANAME): data name to be loaded
            run_id (int): the run_id to be loading the data from

        Return:
            data requested
            MissingDataException if not found
        """

        while True:
            if run_id < 0:
                raise Exception("No instance of '%s' was found in data manager using any run_id." % str(data_name))

            try:
                result = self.get(data_name, run_id)
                logger.info("Data: %s loaded successfully at run: %s. data: %s" % (data_name, str(run_id), str(result)))
                break
            except MissingDataException as e:
                logger.warning("Failed loading data: %s at run_id %s. Exception: %s" % (data_name, str(run_id), e))
                run_id -= 1

        return result


class DataStorage(object):
    @abc.abstractmethod
    def get(self, run_id):
        pass

    @abc.abstractmethod
    def store(self, data, run_id):
        pass


class TFAgentStorage(DataStorage):
    '''This stores a trained agent by calling the persist() function, it returns the path to that agent.
       NOTE: while this handles persisting the agent, the caller is responsible for using the path to load an agent'''

    def __init__(self, rl_app, path_root, suffix=""):
        self.path_storage = PathStorage(path_root, suffix)
        self.path_root = path_root.rstrip('/')
        self.rl_app = rl_app

    def get(self, run_id):
        logger.info("Started Agent Restoration process. run_id: %s" % str(run_id))

        agent = self.rl_app.init_agent()
        assert (isinstance(agent, TFAgent))

        path = self.path_storage.get(run_id)
        try:
            self._restore(agent, path)
        except Exception as e:
            raise MissingDataException("Missing agent from path={} in RLAgentStorage. exception: {}".format(path, str(e)))

        logger.info("Restoration process completed successfully. Checkpoint path: %s" % (str(path)))

        return agent

    @staticmethod
    def _restore(agent, path):
        checkpoint = tf.train.Checkpoint(saved_rl_agent=agent)
        restore_file_path = tf.train.latest_checkpoint(path)
        status = checkpoint.restore(restore_file_path)
        status.assert_existing_objects_matched()

    def store(self, agent, run_id):
        logger.info("Started TF Agent Persistence process.")

        # add checkpoint file name to the path. it is required to add the suffix to the path
        path = self.path_storage.get(run_id)
        checkpoint_file_path = os.path.join(path, "rl_agent_checkpoint")

        checkpoint = tf.train.Checkpoint(saved_rl_agent=agent)
        checkpoint.save(file_prefix=checkpoint_file_path)
        logger.info("Successfully persisted tf agent. Checkpoint path: %s" % checkpoint_file_path)


class ParquetSparkStorage(DataStorage):
    """This stores/retrieves data as Parquet from Spark (hdfs or s3)"""

    def __init__(self, path_root, suffix="", spark=None):
        self.path_storage = PathStorage(path_root, suffix)
        self.path_root = path_root.rstrip('/')
        self.spark = spark
        if self.spark is None:
            self.spark = get_spark_session()

    def get(self, run_id):
        path = self.path_storage.get(run_id)
        try:
            df = self.spark.read.parquet(path)
        except Exception as e:
            raise MissingDataException("Missing data from run_id={} in ParquetSparkStorage. exception: {}".format(run_id,
                                                                                                                  str(e)))
        return df

    def store(self, df, run_id):
        full_path = self.path_storage.get(run_id)
        # Remove the run_id column (if it exists) since this value will be captured by the partition path.
        out_df = df
        out_df.write.parquet(full_path, mode="overwrite")


class S3JSONStorage(DataStorage):
    '''This stores/retrieves data as json'''

    def __init__(self, path_root, suffix=""):
        self.path_storage = PathStorage(path_root, suffix)
        self.path_root = path_root.rstrip('/')
        self.s3_client = boto3.client("s3")

    def get(self, run_id):
        """
        retrieve data stored in s3.

        Return:
             dictionary
             Throw an error if not found.
        """
        path = self.path_storage.get(run_id)
        bucket, key = S3JSONStorage._get_bucket_and_key(path)
        # this throws ClientError with code "NoSuchKey" if file not exist
        try:
            data_obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise MissingDataException("Missing data from run_id={} in S3JSONStorage".format(run_id))
            else:
                raise

        data = eval(data_obj["Body"].read().decode('utf-8'))
        return data

    def store(self, data_dict, run_id):
        """
        store data dictionary in s3 as a json
        """
        path = self.path_storage.get(run_id)
        bucket, key = S3JSONStorage._get_bucket_and_key(path)
        # use repr instead of json.dumps because data_dict could contain tuple
        self.s3_client.put_object(Body=repr(data_dict), Bucket=bucket, Key=key)

    @staticmethod
    def _get_bucket_and_key(path):
        """
        retrieve s3 bucket and key from path. Path format should be: s3[an]://<bucket>/<key>
        """
        m = re.match(r"s3[an]?://(?P<bucket>[^/]+)/(?P<key>.+)", path)
        if m:
            bucket = m.group('bucket')
            key = m.group('key')
        else:
            raise Exception("invalid S3 path: {}".format(path))
        return bucket, key


class InMemoryStorage(DataStorage):
    '''This stores a link to the given data in memory'''

    def __init__(self):
        self.time_to_data = {}

    def get(self, run_id):
        if run_id is None:
            # TODO: could expose union function to join data together
            raise Exception("run_id must be specified for InMemoryStorage")

        data = self.time_to_data.get(run_id)
        if data is None:
            raise MissingDataException("No data previously stored for run_id %s" % run_id)
        return data

    def store(self, data, run_id):
        self.time_to_data[run_id] = data


class PathStorage(DataStorage):
    '''This just stores and returns the path to a resource'''

    def __init__(self, path_root, suffix=""):
        self.path_root = path_root.rstrip('/')
        self.suffix = suffix

    def get(self, run_id):
        return self.path_root + "/run_id=%s/%s" % (run_id, self.suffix)

    def store(self, df, run_id):
        raise Exception("store() not implemented for PathStorage")
