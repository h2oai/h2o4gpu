#!/bin/bash -e
#===============================================================================
# name: daal_preinstalled_packages.sh
#
# installs to the system intel daal libraries and pydaal (python version of daal)
#===============================================================================
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "DIR** $DIR"

check_python_version() { 
	# check if in virtual environment, otherwise it won't work
	if [[ "$VIRTUAL_ENV" != "" ]]
	then
		echo "We are in virtual environment"
		source "$VIRTUAL_ENV/bin/activate"
		ver=$(python --version 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
		if [ "$ver" -lt "35" ]; then
	    	echo "PyDAAL for version 2.x.x is not supported."
	    	exit 1
		fi
	else
	  	echo "Virtual environment required!"
	  	exit 1
	fi
	
}

install_daal_env() { 
	_daal_targz=""
	current_dir=$PWD
	if [  -z "$1" ]; then
		_daal_targz="$DIR/daal.tar.gz"
		echo "** $_daal_targz"
	fi
	
	if [ ! -f $_daal_targz ]; then
		echo "File <daal.tar.gz> not found!"
		exit 1
	fi
	
	mkdir -p $HOME/daal 
	tar -zxvf $_daal_targz -C $HOME
	source $HOME/daal/bin/daalvars.sh
	echo "** Installing PyDAAL wheel"
	source "$VIRTUAL_ENV/bin/activate"
	pip -V
	pip install $HOME/daal/bin/pydaal-*.whl
	echo "** Intel DAAL installed"	
}

check_python_version
install_daal_env "$@"

