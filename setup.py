from setuptools import find_packages
import sys
from distutils.core import setup
from rl_bakery.version import __version__

requirements_path = "./requirements.txt"
with open(requirements_path, 'r') as requirements_file:
    dependency_list = requirements_file.read().splitlines()

print("Loaded the dependencies from requirements.txt: %s" % str(dependency_list))


LIBRARY = "rl_bakery"

setup(name=LIBRARY,
      version=__version__,
      py_modules=[LIBRARY],
      package_dir={
          LIBRARY: LIBRARY,
      },
      packages=find_packages(),
      install_requires=dependency_list
     )
