default: all

all:
	python setup.py clean --all && rm -rf h2oaiglm.egg-info && rm -rf h2oaiglm/__pycache__/
	python setup.py install
	rm -rf dist
	python setup.py sdist bdist_wheel

#	python3 setup.py clean --all && rm -rf h2oaiglm.egg-info && rm -rf h2oaiglm/__pycache__/
#	python3 setup.py install
#	rm -rf dist
#	python3 setup.py sdist bdist_wheel

clean:
	python setup.py clean --all && rm -rf h2oaiglm.egg-info && rm -rf h2oaiglm/__pycache__/ && rm -rf dist/
#	python3 setup.py clean --all && rm -rf h2oaiglm.egg-info && rm -rf h2oaiglm/__pycache__/

