.PHONY: install install-dev install-train make-sample download preprocess split train-baselines train-ranker evaluate serve test lint docker-build docker-run mlflow-ui

PYTHON ?= python
CONFIG ?= configs/retailrocket.yaml
MODEL ?= artifacts/ranker.joblib

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-train:
	$(PYTHON) -m pip install -e ".[train,dev]"

make-sample:
	$(PYTHON) -m recsys.cli make-sample --output data/raw/retailrocket/events.csv

download:
	$(PYTHON) -m recsys.cli download --config $(CONFIG)

preprocess:
	$(PYTHON) -m recsys.cli preprocess --config $(CONFIG)

split:
	$(PYTHON) -m recsys.cli split --config $(CONFIG)

train-baselines:
	$(PYTHON) -m recsys.cli train-baselines --config $(CONFIG)

train-ranker:
	$(PYTHON) -m recsys.cli train-ranker --config $(CONFIG)

evaluate:
	$(PYTHON) -m recsys.cli evaluate --config $(CONFIG) --model $(MODEL)

serve:
	$(PYTHON) -m uvicorn recsys.api.main:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests

docker-build:
	docker build -t ecommerce-recsys-ranking-service .

docker-run:
	docker run --rm -p 8000:8000 ecommerce-recsys-ranking-service

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
