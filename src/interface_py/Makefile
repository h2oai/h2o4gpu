SHELL := /bin/bash # force avoidance of dash as shell
thepwd = "$(shell pwd)"

default: all

pylint:
	@status=0; \
	for py in $$(find h2o4gpu -name "*.py" -type f); do \
		if [ "$$py" == "h2o4gpu/__init__.py" ]; \
		then echo "Skip $$py"; \
		else echo $$py; \
		mkdir -p ./pylint.d ; PYLINTHOME=./pylint.d pylint -j 16 --rcfile=../../tools/pylintrc -rn $$py; \
		tmp=$$?; \
		if [ $$tmp -ne 0 ]; then status=$$tmp; fi; \
		fi; \
	done; \
	exit $$status

pyformat:
	@status=0; \
	for py in $$(find h2o4gpu -name "*.py" -type f); do \
		if [ "$$py" == "h2o4gpu/__init__.py" ]; \
		then echo "Skip $$py"; \
		else echo $$py; \
		yapf -p -i --style google $$py; \
		fi; \
	done; \

prep:
	PYVER=`python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' | sed 's/\.//g'` && sed -i 's/python-tag=.*/python-tag=py'$$PYVER'/g' setup.cfg
	PYVER=`python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'` && sed -i 's/python_version==.*/python_version=='$$PYVER'/g' setup.cfg

all: prep

	#cat ../../requirements_buildonly.txt ../../requirements_runtime.txt > requirements.txt
	#cat ../../requirements_runtime.txt > requirements.txt

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
	-cd ../.. && pip uninstall -y h2o4gpu
	python setup.py install

#	python3 setup.py clean --all && rm -rf h2o4gpu.egg-info && rm -rf h2o4gpu/__pycache__/
#	python3 setup.py install
#	rm -rf dist
#	python3 setup.py sdist bdist_wheel

clean:
	python setup.py clean --all && rm -rf h2o4gpu.egg-info && rm -rf h2o4gpu/__pycache__/ && rm -rf dist/
	cd h2o4gpu && find -L -name *.pyc | xargs rm -rf


