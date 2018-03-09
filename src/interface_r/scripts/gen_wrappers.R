# Instruction:
# * Ensure you have the following R libraries installed via the following:
#   `devtools::install_github("rstudio/reticulate"); install.packages(c("stringr", "roxygen2))`
# * Change variable `pkg_location` below to point to your R package directory
# * Run `use_condaenv()`, `use_python()`, or `use_virtualenv()` from reticulate to activate the
#   environment with h2o4gpu installed. 
#   e.g. use_condaenv("r-h2o4gpu", conda = "~/miniconda3/envs/py36/bin/conda")
library(reticulate)
#use_virtual_env() or use_condaenv() or use_python()
library(stringr)
library(roxygen2)
pkg_location <- "~/h2o4gpu/src/interface_r"


# Note: Code below should not be changed unless necessary

source(file.path(pkg_location, "scripts/gen_wrapper_utils.R"))
h2o4gpu <- import("h2o4gpu")
common_nullabl_int_params <- c("random_state")

file_name <- "R/auto_generated_wrappers.R"
write("# -------------------------------------------------------------------------------
#             *** WARNING: DO NOT MODIFY THIS FILE ***
#              
# Instead, modify scripts/gen_wrappers.R which automatically generates this file
# ------------------------------------------------------------------------------- \n", file = file_name)

test_script_file_name <- "tests/testthat/test_auto_generated_wrappers.R"
write("# -------------------------------------------------------------------------------
#             *** WARNING: DO NOT MODIFY THIS FILE ***
#              
# Instead, modify scripts/gen_wrappers.R which automatically generates this file
# ------------------------------------------------------------------------------- \n", file = test_script_file_name)

write(
  'context("Test model wrappers")\nsource("helper-utils.R")\n',
  file = test_script_file_name,
  append = TRUE
)

# Supervised models

write_wrapper("h2o4gpu$RandomForestClassifier",
            r_function = "h2o4gpu.random_forest_classifier",
            nullable_int_params = common_nullabl_int_params,
            class_tags = 'c("classifier")',
            file_name = file_name,
            test_script_file_name = test_script_file_name,
            description = "Random Forest Classifier")

write_wrapper("h2o4gpu$RandomForestRegressor",
            r_function = "h2o4gpu.random_forest_regressor",
            nullable_int_params = common_nullabl_int_params,
            class_tags = 'c("regressor")',
            file_name = file_name,
            test_script_file_name = test_script_file_name,
            description = "Random Forest Regressor")

write_wrapper("h2o4gpu$GradientBoostingClassifier",
            r_function = "h2o4gpu.gradient_boosting_classifier",
            nullable_int_params = common_nullabl_int_params,
            class_tags = 'c("classifier")',
            file_name = file_name,
            test_script_file_name = test_script_file_name,
            description = "Gradient Boosting Classifier")

write_wrapper("h2o4gpu$GradientBoostingRegressor",
            r_function = "h2o4gpu.gradient_boosting_regressor",
            nullable_int_params = common_nullabl_int_params,
            class_tags = 'c("regressor")',
            file_name = file_name,
            test_script_file_name = test_script_file_name,
            description = "Gradient Boosting Regressor")

write_wrapper("h2o4gpu$ElasticNet",
            r_function = "h2o4gpu.elastic_net_regressor",
            nullable_int_params = common_nullabl_int_params,
            class_tags = 'c("regressor")',
            file_name = file_name,
            test_script_file_name = test_script_file_name,
            description = "Elastic Net Regressor")

write_wrapper("h2o4gpu$ElasticNet",
              r_function = "h2o4gpu.elastic_net_classifier",
              nullable_int_params = common_nullabl_int_params,
              class_tags = 'c("classifier")',
              file_name = file_name,
              test_script_file_name = test_script_file_name,
              description = "Elastic Net Classifier")

# Unsupervised models

write_wrapper("h2o4gpu$KMeans",
            r_function = "h2o4gpu.kmeans",
            nullable_int_params = common_nullabl_int_params,
            class_tags = "NULL",
            file_name = file_name,
            test_script_file_name = test_script_file_name,
            description = "K-means Clustering")

write_wrapper("h2o4gpu$PCA",
            r_function = "h2o4gpu.pca",
            nullable_int_params = common_nullabl_int_params,
            class_tags = "NULL",
            file_name = file_name,
            test_script_file_name = test_script_file_name,
            description = "Principal Component Analysis (PCA)")

write_wrapper("h2o4gpu$TruncatedSVD",
            r_function = "h2o4gpu.truncated_svd",
            nullable_int_params = common_nullabl_int_params,
            class_tags = "NULL",
            file_name = file_name,
            test_script_file_name = test_script_file_name,
            description = "Truncated Singular Value Decomposition (TruncatedSVD)")

# Regenerate NAMESPACE and .Rd files
roxygen2::roxygenise(pkg_location)
