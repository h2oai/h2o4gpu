# -------------------------------------------------------------------------------
#             *** WARNING: DO NOT MODIFY THIS FILE ***
#              
# Instead, modify scripts/gen_wrappers.R which automatically generates this file
# ------------------------------------------------------------------------------- 

context("Test model wrappers")
source("helper-utils.R")

test_classifier(h2o4gpu.random_forest_classifier, "h2o4gpu.random_forest_classifier")
test_regressor(h2o4gpu.random_forest_regressor, "h2o4gpu.random_forest_regressor")
test_classifier(h2o4gpu.gradient_boosting_classifier, "h2o4gpu.gradient_boosting_classifier")
test_regressor(h2o4gpu.gradient_boosting_regressor, "h2o4gpu.gradient_boosting_regressor")
test_regressor(h2o4gpu.elastic_net_regressor, "h2o4gpu.elastic_net_regressor")
test_classifier(h2o4gpu.elastic_net_classifier, "h2o4gpu.elastic_net_classifier")
test_unsupervised(h2o4gpu.kmeans, "h2o4gpu.kmeans")
test_unsupervised(h2o4gpu.pca, "h2o4gpu.pca")
test_unsupervised(h2o4gpu.truncated_svd, "h2o4gpu.truncated_svd")
