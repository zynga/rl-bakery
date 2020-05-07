from datetime import datetime, timedelta
from rl_bakery.engine.abstract_engine_config import MockEngineConfig
from unittest import TestCase


class TestAbstractEngineConfig(TestCase):

    def test_get_run_id_1(self):
        mock_engine_config = MockEngineConfig()
        mock_engine_config.training_interval = timedelta(days=1)
        mock_engine_config.start_dt = datetime(2019, 11, 2, 0, 0, 0)
        MockEngineConfig._get_current_datetime = lambda _: datetime(2019, 11, 2, 10, 0, 0)

        res = mock_engine_config.get_current_run_id()

        expected_res = 0
        self.assertEquals(res, expected_res)

    def test_get_run_id_2(self):
        mock_engine_config = MockEngineConfig()
        mock_engine_config.start_dt = datetime(2019, 11, 2, 0, 0, 0)
        mock_engine_config.training_interval = timedelta(hours=4)
        MockEngineConfig._get_current_datetime = lambda _: datetime(2019, 11, 3, 0, 0, 0)

        res = mock_engine_config.get_current_run_id()

        expected_res = 6
        self.assertEquals(res, expected_res)
