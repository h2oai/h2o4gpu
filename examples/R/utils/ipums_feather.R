## First time only: install both packages
install.packages("data.table", type = "source", repos = "http://Rdatatable.github.io/data.table",dependencies=TRUE)
devtools::install_github("wesm/feather/R")
require(data.table)   
require(feather)
DT <- fread("./ipums.csv")
write_feather(DT, "ipums.feather")
