context("Test model wrappers")

source("helper-utils.R")



test_classifier(h2o4gpu.random_forest_classifier, "h2o4gpu.random_forest_classifier")
test_classifier(h2o4gpu.gradient_boosting_classifier, "h2o4gpu.gradient_boosting_classifier")

test_regressor(h2o4gpu.random_forest_regressor, "h2o4gpu.random_forest_regressor")
test_regressor(h2o4gpu.gradient_boosting_regressor, "h2o4gpu.gradient_boosting_regressor")

test_unsupervised(h2o4gpu.kmeans, "h2o4gpu.kmeans")
