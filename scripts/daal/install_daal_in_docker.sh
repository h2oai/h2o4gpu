#!/bin/bash
#===============================================================================
# name: install_daal_in_docker.sh
#
# installs to the system intel daal libraries and pydaal (python version of daal)
#===============================================================================
_indel_dall_tar="https://s3.amazonaws.com/intel-daal/daal-linux_x86_64.tar.gz"
_daal_root="$HOME/daal"
_pwd=$PWD

apt-get install axel -y
mkdir -p $_daal_root
cd /tmp/ && axel -a -n 20 $_indel_dall_tar && tar -xzvf daal-linux_x86_64.tar.gz -C $HOME
echo "** Successfully downloaded DAAL. **"
cd $_daal_root && pip install pydaal-2018*.whl && export LD_LIBRARY_PATH=$_daal_root/lib:$LD_LIBRARY_PATH && cd $PWD
cd $_pwd
echo "** The DAAL environment is all set up. **"
