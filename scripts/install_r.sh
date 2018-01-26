#!/bin/bash

set -e

R_VERSION=$1
R_VERSION_HOME="/usr/local/R"

# Create temporary directory
mkdir -p $HOME/R_tmp
cd $HOME/R_tmp

# Download and extract R source
echo Downloading R source
wget https://cran.r-project.org/src/base/R-3/R-${R_VERSION}.tar.gz
tar xvzf R-${R_VERSION}.tar.gz
rm R-${R_VERSION}.tar.gz

# Configure and make
cd R-${R_VERSION}
./configure --prefix=${R_VERSION_HOME} --with-x=no --enable-utf
make
make install
chmod a+w -R ${R_VERSION_HOME}/lib/R/library

# Cleanup
cd ../..
rm -rf $HOME/R_tmp

# Create symbolic link
ln -s ${R_VERSION_HOME}/bin/R /usr/bin/R
ln -s ${R_VERSION_HOME}/bin/Rscript /usr/bin/Rscript
echo The R-${R_VERSION} executable is now available in /usr/bin/R

# Create dirs for current R and make them writable
mkdir -p /usr/local/R/current/bin/
chmod a+w /usr/local/R/current/bin/

echo "Generate activate script"
# Create activation script
echo """#! /bin/bash
ln -s -f ${R_VERSION_HOME}/bin/R /usr/local/R/current/bin/R
ln -s -f ${R_VERSION_HOME}/bin/Rscript /usr/local/R/current/bin/Rscript""" > /usr/bin/activate_R_${R_VERSION}
chmod a+x /usr/bin/activate_R_${R_VERSION}

# Activate this R version
echo "Activating R ${R_VERSION}"
activate_R_${R_VERSION}
