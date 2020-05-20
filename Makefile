setup:
	python3 -m venv .

install:
	pip install --upgrade pip &&\
        pip install -r requirements.txt

test:
	python3 -m unittest discover

build_wheel:
	python3 setup.py bdist_wheel

lint:
	flake8 rl_bakery

all: install lint test
