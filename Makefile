.PHONY: install data run test all clean
install:
	pip install -r requirements.txt pytest
data:
	python src/generate_data.py
run:
	python src/pipeline.py && python src/build_readme.py && python src/build_dashboard.py
test:
	pytest -q
all: data run test
clean:
	rm -rf data/processed outputs reports/figures/*.png dashboard/*.csv dashboard/credit_risk_dashboard.html _site
