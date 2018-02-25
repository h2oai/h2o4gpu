#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

R_BIN=$(which R)
if [[ $? != 0 ]]; then
  echo "R is not installed. Skipping R tests..."
else
  echo "R is installed. Running R tests..."
  R -e 'pkgs <- c("devtools", "magrittr", "roxygen2", "reticulate", "testthat"); if (isTRUE(all(pkgs %in% row.names(installed.packages())))) devtools::test("src/interface_r")'
fi
