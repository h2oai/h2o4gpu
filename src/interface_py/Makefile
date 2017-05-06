default: all

all:
	python setup.py clean --all && rm -rf h2oaiglm.egg-info && rm -rf h2oaiglm/__pycache__/
	python setup.py install --user
	rm -rf dist
	python setup.py sdist bdist_wheel

clean:
	python setup.py clean --all && rm -rf h2oaiglm.egg-info && rm -rf h2oaiglm/__pycache__/

