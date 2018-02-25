#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

R_BIN=$(which R)
if [[ $? != 0 ]]; then
  echo "R is not installed. Installing..."
  $DIR/install_r.sh 3.1.0
else
  echo "R is installed."
fi

echo "Installing necessary R packages..."
R -e 'options(repos = c(CRAN = "http://cran.rstudio.com")); pkgs <- c("devtools", "magrittr", "roxygen2"); pkgs_to_install <- pkgs[!pkgs %in% row.names(installed.packages())]; if (length(pkgs_to_install) != 0) install.packages(pkgs_to_install); if (!require("reticulate")) devtools::install_github("rstudio/reticulate"); if (!require("testthat")) devtools::install_github("r-lib/testthat@v1.0.2")'
R -e 'devtools::install_local("src/interface_r/")'
