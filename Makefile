setup:
	python3 -m venv .

install:
	pip install --upgrade pip &&\
        pip install -r requirements.txt

test:
	python3 -m unittest discover

test-integration:
	bash integration_test/run_integration_tests.sh

build_wheel:
	python3 setup.py bdist_wheel

lint:
	- flake8 rl_bakery
	- flake8 integration_test

all: install lint test
