setup:
	python setup.py

train:
	python pipeline/pipeline.py

retrain:
	python pipeline/pipeline.py --retrain

trackapp:
	mlflow ui
