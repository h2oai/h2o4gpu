#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Installing necessary apt-get dependencies..."
apt-get update -y
apt-get -y install libcurl4-openssl-dev libssl-dev libxml2-dev

R_BIN=$(which R)
if [[ $? != 0 ]]; then
  echo "R is not installed. Installing..."
  $DIR/install_r.sh 3.1.0
else
  echo "R is installed."
fi

echo "Installing necessary R packages..."
R -e 'options(repos = c(CRAN = "http://cran.rstudio.com")); pkgs <- c("devtools", "testthat", "magrittr", "roxygen2"); pkgs_to_install <- pkgs[!pkgs %in% row.names(installed.packages())]; if (length(pkgs_to_install) != 0) install.packages(pkgs_to_install); if (!require("reticulate")) devtools::install_github("rstudio/reticulate")'
R -e 'devtools::install_local("src/interface_r/")'
