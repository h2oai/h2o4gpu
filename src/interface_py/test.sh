#!/bin/sh

status=0
for py in `find h2o4gpu -name "*.py" -type f`; do 
	pylint --rcfile=../../tools/pylintrc -rn $py; 
	tmp=$?
	if [ $tmp -ne 0 ]; then
		echo $status
		status=$tmp
	fi
done
exit $status
