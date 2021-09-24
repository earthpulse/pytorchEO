pypi: test dist
	twine upload --repository pypi dist/*
dist: clean
	python setup.py sdist
clean:
	rm -rf dist
docum: 
	python sphinx/rst.py pytorch_eo sphinx/source
	python setup.py build_sphinx --build-dir sphinx/build
	rm -rf docs
	mv sphinx/build/html docs
test:
	pytest tests