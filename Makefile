.PHONY: run test lint dockerize clean

run:
	python app.py

run-cli:
	python main.py

test:
	pytest

lint:
	flake8 .
	black --check .

format:
	black .

dockerize:
	docker build -t smart-trade-advisor:latest .

docker-run:
	docker run -p 5000:5000 smart-trade-advisor:latest

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete 