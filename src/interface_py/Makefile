location = $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
WHERE := $(location)
$(info ** -> $(WHERE))
$(info ** ------------------------------------------------------------------ **)

SHELL := /bin/bash # force avoidance of dash as shell
thepwd = "$(shell pwd)"

include ../../make/config.mk

default: all

pylint:
	$$SHELL scripts/run-pylint.sh

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
	PYVER=`$(PYTHON) -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' | sed 's/\.//g'` && sed -i 's/python-tag=.*/python-tag=py'$$PYVER'/g' setup.cfg
	PYVER=`$(PYTHON) -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'` && sed -i 's/python_version==.*/python_version=='$$PYVER'/g' setup.cfg
	cd ../../ && bash scripts/apply_sklearn_initmerge.sh # so if change our init it gets updated

all: prep

	# for incorporating xgboost and py3nvml and lightgbm builds
	rm -rf ./xgboost
	rm -rf ./xgboost_prev
	ln -sf ../../xgboost/python-package/xgboost ./xgboost
	ln -sf ../../xgboost_prev/python-package/xgboost ./xgboost_prev
	rm -rf ../../xgboost/python-package/xgboost/src/
	rm -rf ../../xgboost_prev/python-package/xgboost/src/
	ln -sf ../../LightGBM/python-package/dist_gpu/lightgbm lightgbm_gpu
	ln -sf ../../LightGBM/python-package/dist_cpu/lightgbm lightgbm_cpu
	ln -sf ../../py3nvml/py3nvml .

	$(PYTHON) setup.py sdist bdist_wheel

	# update build with lightgbm shared library
	if [ `arch` != "ppc64le" ]; then \
	mkdir -p build/lib/lightgbm_gpu/ ; cd build/lib/lightgbm_gpu/ ; ln -sf ../../../../../LightGBM/python-package/dist_gpu/lightgbm/lib_lightgbm.so . ; cd ../../../ ; \
	fi
	mkdir -p build/lib/lightgbm/ ; cd build/lib/lightgbm_cpu/ ; ln -sf ../../../../../LightGBM/python-package/dist_cpu/lightgbm/lib_lightgbm.so . ; cd ../../../ 

	# Make wheel with other builds added
	rm -rf dist/*.whl
	$(PYTHON) setup.py sdist bdist_wheel

	# for pycharm
	ln -sf $(thepwd)/../interface_c/_ch2o4gpu_cpu.so .
	ln -sf $(thepwd)/../interface_c/_ch2o4gpu_gpu.so .


install:
	-cd ../.. && $(PYTHON) -m pip uninstall -y h2o4gpu
	-cd ../.. && $(PYTHON) -m pip uninstall -y xgboost
	-cd ../.. && $(PYTHON) -m pip uninstall -y py3nvml
	-cd ../.. && $(PYTHON) -m pip uninstall -y lightgbm
	find -name *.pyc | xargs rm -rf
	find -name *__pycache__ | xargs rm -rf
	# something wrong with below
	#$(PYTHON) setup.py install
	$(PYTHON) -m pip install dist/h2o4gpu*.whl --upgrade --constraint requirements_buildonly.txt

clean:
	rm -f h2o4gpu/BUILD_INFO.txt
	rm -f h2o4gpu/dist*
    # A bit hacky b/c of how we are merging __init__.py with other frameworks
    # This *is* required
	rm -f h2o4gpu/__init__.py
	touch h2o4gpu/__init__.py
	$(PYTHON) setup.py clean --all && rm -rf h2o4gpu.egg-info && rm -rf h2o4gpu/__pycache__/ && rm -rf dist/
	cd h2o4gpu && find -L -name *.pyc | xargs rm -rf


