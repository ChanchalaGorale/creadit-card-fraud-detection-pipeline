setup:
	python setup.py

train:
	python pipeline/pipeline.py

retrain:
	python pipeline/pipeline_retrain.py

trackapp:
	mlflow ui
