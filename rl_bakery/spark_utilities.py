import unittest
from pyspark.sql.session import SparkSession
from pyspark import SparkConf
import logging
import math
from datetime import datetime


def check_df_value(out_df, expected, unique_row_ids, float_tolerance=float(0.00001)):
    """
    Compares a Spark Dataframe to a set of expected values, raising exceptions on detected differences.

    Args:
    out_df: A Spark Dataframe to check.
    expected: The expected values in the out_df. This is a list of rows, each row is a dictionary of column name to
        value.
    unique_row_ids: The columns that are unique to each row. These are used to match rows between out_df and expected.
    float_tolerance: The amount of acceptable difference between float values.
    """

    def extract_row_key(row, unique_row_ids):
        row_ids = []
        for k in unique_row_ids:
            if k not in row:
                raise Exception("Missing row id %s from output %s" % (k, row))
            row_ids.append(row[k])
        return row_ids

    out = out_df.collect()

    out_map = {}
    for r in out:
        k = extract_row_key(r, unique_row_ids)
        out_map[str(k)] = r

    for r in expected:
        k = extract_row_key(r, unique_row_ids)
        if str(k) not in out_map:
            raise Exception("Missing data for user %s" % k)

        out_row = out_map[str(k)]
        for attribute, value in r.items():
            if attribute not in out_row:
                raise Exception("Missing expected attribute %s from row %s" % (attribute, out_row))

            if type(out_row[attribute]) != type(value):
                raise Exception("Type mismatch for attribute %s for row %s is %s, expecting %s" % (
                    attribute, out_row, out_row[attribute], value))

            if isinstance(out_row[attribute], float):
                if not math.isclose(out_row[attribute], value, abs_tol=float_tolerance):
                    raise Exception(
                        "Attribute %s for row %s was %s, expected %s" % (attribute, out_row, out_row[attribute], value))
            elif isinstance(out_row[attribute], datetime) and out_row[attribute].tzinfo != value.tzinfo:
                input_value = out_row[attribute].replace(tzinfo=value.tzinfo)
                if input_value != value:
                    raise Exception(
                        "datetime attribute %s for row %s was %s, expected %s" % (
                            attribute, out_row, input_value, value))
            else:
                if out_row[attribute] != value:
                    raise Exception(
                        "Attribute %s for row %s was %s, expected %s" % (attribute, out_row, out_row[attribute], value))


def compare_dataframes(df1, df2):
    """
    Compare two dataframes and

    Returns:
         Boolean: return True if they are similar, False otherwise
         string: error message containing the different keys
    """
    if set(df1.columns) != set(df2.columns):
        return False

    # make sure columns are ordered the same way
    df2 = df2.select(df1.columns)

    diff1 = df1.subtract(df2)
    diff2 = df2.subtract(df1)

    if diff1.count() > 0 or diff2.count() > 0:
        return False
    else:
        return True


def get_spark_session():
    if in_spark_env():
        # The spark variable representing a Spark session is already available
        return spark  # noqa: F821
    else:
        conf = SparkConf().set('spark.driver.host', '127.0.0.1')
        return SparkSession.builder.master('local'). \
            appName("local-pyspark-test-context"). \
            config(conf=conf). \
            getOrCreate()


def in_spark_env():
    if 'spark' in locals():
        return True
    else:
        return False


class PySparkTestCase(unittest.TestCase):
    """
    This unit test adds an attribute self.spark to the TestCase. It detects whether the spark variable is available
    locally otherwise it creates a local Spark Session
    """

    @classmethod
    def suppress_py4j_logging(cls):
        logger = logging.getLogger('py4j')
        logger.setLevel(logging.WARN)

    @classmethod
    def get_spark_session(cls):
        return get_spark_session()

    @classmethod
    def cleanup_spark_session(cls):
        if cls.spark and not in_spark_env():
            cls.spark.stop()
            cls.spark = None

    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.get_spark_session()

    @classmethod
    def teardownClass(cls):
        cls.cleanup_spark_session()


def get_field_type_from_dataframe(df, field_name):
    """
    """
    df_schema = df.schema
    for struct_field in df_schema:
        if struct_field.name == field_name:
            return struct_field.dataType

    raise Exception("Field %s not found in schema: %s" % (field_name, str(df_schema)))
