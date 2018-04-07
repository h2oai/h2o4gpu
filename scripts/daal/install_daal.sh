#!/bin/bash
#===============================================================================
# name: install_daal.sh
#
# installs to the system intel daal libraries and pydaal (python version of daal)
#===============================================================================
set -e

_intel_dall_tar="https://s3.amazonaws.com/intel-daal/daal-linux_x86_64__cp36.tar.gz"

if hash axel 2>/dev/null; then
    axel -a -n 20 $_intel_dall_tar
else
    wget $_intel_dall_tar
fi

tar xzvf daal-linux_x86_64__cp36.tar.gz -C $HOME
rm -rf daal-linux_x86_64__cp36.tar.gz

pip install $HOME/daal/pydaal-2018.0.1.20171012-cp36-none-linux_x86_64.whl
ln -sf $HOME/daal/lib/libtbb.so.2 /usr/lib/libtbb.so.2
ln -sf $HOME/daal/lib/libtbb.so /usr/lib/libtbb.so
ln -sf $HOME/daal/lib/libtbbmalloc.so.2 /usr/lib/libtbbmalloc.so.2
ln -sf $HOME/daal/lib/libtbbmalloc.so /usr/lib/libtbbmalloc.so
ln -sf $HOME/daal/lib/libdaal_sequential.so /usr/lib/libdaal_sequential.so
ln -sf $HOME/daal/lib/libdaal_core.so /usr/lib/libdaal_core.so
ln -sf $HOME/daal/lib/libdaal_thread.so /usr/lib/libdaal_thread.so
