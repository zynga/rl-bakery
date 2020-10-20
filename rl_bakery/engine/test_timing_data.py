from datetime import datetime, timedelta
from rl_bakery.engine.timing_data import TimingData
from unittest import TestCase


class TestTimingData(TestCase):

    def test_get_run_id_1(self):
        td = TimingData(start_dt=datetime(2019, 11, 2, 0, 0, 0),
                        training_interval=timedelta(days=1))
        td._get_current_datetime = lambda : datetime(2019, 11, 2, 10, 0, 0)
        res = td.get_current_run_id()

        expected_res = 0
        self.assertEquals(res, expected_res)

    def test_get_run_id_2(self):
        td = TimingData(start_dt=datetime(2019, 11, 2, 0, 0, 0),
                        training_interval=timedelta(hours=4))
        td._get_current_datetime = lambda : datetime(2019, 11, 3, 0, 0, 0)

        res = td.get_current_run_id()

        expected_res = 6
        self.assertEquals(res, expected_res)
