#!/bin/bash -e
#===============================================================================
# name: daal_preinstalled_packages.sh
#
# installs to the system intel daal libraries and pydaal (python version of daal)
#===============================================================================
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "DIR** $DIR"

_pwd=$PWD
_indel_dall_tar="https://s3.amazonaws.com/intel-daal/daal-linux_x86_64.tar.gz"
_daal_root="$HOME/daal"

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

download_intel_daal_tar() {
	echo "Download"
	# multithreaded download
	if [ $(dpkg-query -W -f='${Status}' axel 2>/dev/null | grep -c "ok installed") -eq 0 ];
	then
		apt-get install axel -y;
	fi
	cd /tmp/ && axel -a -n 20 $_indel_dall_tar && tar -xzvf daal-linux_x86_64.tar.gz -C $HOME	
	echo "For permanent Intel DAAL setup add to your .bashrc or .profile: LD_LIBRARY_PATH=$_daal_root/lib:\$LD_LIBRARY_PATH"
	cd $_daal_root && pip install pydaal-2018*.whl && export LD_LIBRARY_PATH=$_daal_root/lib:$LD_LIBRARY_PATH && cd $PWD
	echo "Your environment is all set up for Intel DAAL"
}


check_python_version
download_intel_daal_tar

