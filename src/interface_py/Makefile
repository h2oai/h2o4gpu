thepwd = "$(shell pwd)"

default: all

all:

	# for incorporating xgboost and py3nvml builds
	ln -sf ../../xgboost/python-package/xgboost .
	ln -sf ../../py3nvml/py3nvml .

	python setup.py sdist bdist_wheel

	# update build with xgboost shared library
	mkdir -p build/lib/xgboost/
	cd build/lib/xgboost/ ; ln -sf ../../../../../xgboost/lib/libxgboost.so  . ; cd ../../../

	# Make wheel with other builds added
	rm -rf dist/*.whl
	python setup.py sdist bdist_wheel

	# for pycharm
	ln -sf $(thepwd)/build/lib/ch2o4gpu_cpu.so .
	ln -sf $(thepwd)/build/lib/ch2o4gpu_gpu.so .

install:
	python setup.py install

#	python3 setup.py clean --all && rm -rf h2o4gpu.egg-info && rm -rf h2o4gpu/__pycache__/
#	python3 setup.py install
#	rm -rf dist
#	python3 setup.py sdist bdist_wheel

clean:
	python setup.py clean --all && rm -rf h2o4gpu.egg-info && rm -rf h2o4gpu/__pycache__/ && rm -rf dist/
#	python3 setup.py clean --all && rm -rf h2o4gpu.egg-info && rm -rf h2o4gpu/__pycache__/

