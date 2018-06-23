#!/bin/bash

exec 3>&1

status=0;
for py in `find h2o4gpu -name "*.py" -type f`; do
	if 	[ "$py" == "h2o4gpu/__init__.py" ] || [ "$py" == "h2o4gpu/util/roc_opt.py" ] || [ "$py" == "h2o4gpu/solvers/daal_solver/utils/__init__.py" ] || [ "$py" == "h2o4gpu/solvers/__init__.py" ] || [ "$py" == "h2o4gpu/typecheck/compatibility.py" ] || [ "$py" == "h2o4gpu/typecheck/typechecks.py" ] || [ "$py" == "h2o4gpu/libs/ch2o4gpu_gpu.py" ] || [ "$py" == "h2o4gpu/libs/ch2o4gpu_cpu.py" ] || [ "$py" == "h2o4gpu/util/daicx.py" ]; then
		echo "Skip $py"; 
	else echo $py; 
	mkdir -p ../pylint.d ;
	stderr="$( PYLINTHOME=../pylint.d pylint --rcfile=../../tools/pylintrc -rn $py < /dev/stdin 2>&1 1>&3)";
	tmp=$?;
		if [ $tmp -ne 0 ]; then
			echo "STATUS=$?, STDERR: $stderr, file: $py"
			status=$tmp; 
			exit $status
		fi; 
	fi; 
done; 
rm -rf ../pylint.d; 
exit $status
