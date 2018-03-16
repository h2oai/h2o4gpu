#!/bin/bash
#===============================================================================
# name: install_daal_locally.sh
#
# installs to the system intel daal libraries and pydaal (python version of daal)
#===============================================================================
set -e

_intel_dall_tar="https://s3.amazonaws.com/intel-daal/daal-linux_x86_64__cp36.tar.gz"

function daal_downloaded {
	if [ -f "$HOME/daal/pydaal-2018.0.1.20171012-cp36-none-linux_x86_64.whl" ]; then
		echo "PyDAAL wheel already downloaded";
	else
		echo "PyDAAL wheel must be downloaded, this may take a while.";
	fi
}

function pip_wheel {
	echo "Installing PyDAAL ..."
	pip install $HOME/daal/pydaal-2018.0.1.20171012-cp36-none-linux_x86_64.whl &&
	sudo ln -sf $HOME/daal/lib/libtbb.so.2 /usr/lib/libtbb.so.2 &&
	sudo ln -sf $HOME/daal/lib/libtbb.so /usr/lib/libtbb.so &&
	sudo ln -sf $HOME/daal/lib/libtbbmalloc.so.2 /usr/lib/libtbbmalloc.so.2 &&
	sudo ln -sf $HOME/daal/lib/libtbbmalloc.so /usr/lib/libtbbmalloc.so &&
	sudo ln -sf $HOME/daal/lib/libdaal_sequential.so /usr/lib/libdaal_sequential.so &&
	sudo ln -sf $HOME/daal/lib/libdaal_core.so /usr/lib/libdaal_core.so &&
	sudo ln -sf $HOME/daal/lib/libdaal_thread.so /usr/lib/libdaal_thread.so
}

function install_daal {
	echo "Unpacking PyDAAL wheel ..."
	tar xzvf daal-linux_x86_64__cp36.tar.gz -C $HOME &&
	rm -rf daal-linux_x86_64__cp36.tar.gz &&
	eval "$(/root/.pyenv/bin/pyenv init -)" && 
	pip_wheel
}

# detect if axel is installed
daal_downloaded
if [[ $? -ne 0 ]]; then
	if hash axel 2>/dev/null; then
		axel -a -n 20 $_intel_dall_tar && install_daal
	else
		wget $_intel_dall_tar && install_daal
	fi
else
	pip_wheel
fi
