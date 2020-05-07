__path__ = __import__('pkgutil').extend_path(__path__, __name__)


def run_tests():
    import unittest
    import os
    here = os.path.abspath(os.path.dirname(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(here)
    unittest.TextTestRunner().run(suite)
