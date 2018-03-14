## h2o4gpu - R Interface to H2O4GPU

This directory contains the R package for H2O4GPU - a collection of GPU solvers by H2Oai.

### Installation

First, please follow the instruction [here](https://github.com/h2oai/h2o4gpu#installation) to build the H2O4GPU Python package.

Then install `h2o4gpu` R package via the following:

``` r
if (!require(devtools)) install.packages("devtools")
devtools::install_github("h2oai/h2o4gpu", subdir = "src/interface_r")
```

To test your installation, try the following example that builds a simple [XGBoost](https://github.com/dmlc/xgboost) random forest classifier:

``` r
require(h2o4gpu)

# Setup dataset
x <- iris[1:4]
y <- as.integer(iris$Species) - 1

# Initialize and train the classifier
model <- h2o4gpu.random_forest_classifier() %>% fit(x, y)

# Make predictions
predictions <- model %>% predict(x)
```

For more examples, please visit the package [vignettes](https://github.com/h2oai/h2o4gpu/tree/master/src/interface_r/vignettes).
