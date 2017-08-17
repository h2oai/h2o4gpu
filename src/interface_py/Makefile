thepwd = "$(shell pwd)"

default: all

all:
	python setup.py clean --all && rm -rf h2ogpuml.egg-info && rm -rf h2ogpuml/__pycache__/
	python setup.py install
	rm -rf dist

    # for incorporating xgboost
	ln -sf ../../../xgboost/python-package/xgboost .

    mkdir -p build/lib/xgboost/
    cd build/lib/xgboost/
	ln -sf ../../../../../xgboost/lib/libxgboost.so  .
	cd ../../../

    # Make wheel
	python setup.py sdist bdist_wheel
    # for pycharm
	ln -sf $(thepwd)/build/lib/ch2ogpuml_cpu.so .
	ln -sf $(thepwd)/build/lib/ch2ogpuml_gpu.so .

#	python3 setup.py clean --all && rm -rf h2ogpuml.egg-info && rm -rf h2ogpuml/__pycache__/
#	python3 setup.py install
#	rm -rf dist
#	python3 setup.py sdist bdist_wheel

clean:
	python setup.py clean --all && rm -rf h2ogpuml.egg-info && rm -rf h2ogpuml/__pycache__/ && rm -rf dist/
#	python3 setup.py clean --all && rm -rf h2ogpuml.egg-info && rm -rf h2ogpuml/__pycache__/

