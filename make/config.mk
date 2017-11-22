#
# Build configuration
#

# Location of artifacts
# E.g. "s3://bucket/dirname"
ARTIFACTS_BUCKET = s3://artifacts.h2o.ai/releases
# Location of local directory with dependencies
DEPS_DIR = deps

# Location of datasets
SMALLDATA_BUCKET = s3://h2o-public-test-data/smalldata

# Location of other datasets that we add for h2o4gpu testing (ipums, bnp, etc)
DATA_BUCKET = s3://h2o-datasets/h2o4gpu/data

# Location of local directory with data
DATA_DIR =

# Location of open data
OPEN_DATA_BUCKET = s3://h2o-public-test-data/h2o4gpu/open_data

# Location of local directory with open data
OPEN_DATA_DIR = open_data

#
# PyDataTable
#
#PYDATATABLE_VERSION = 0.1.0+master.97


#
# XGBoost
#
XGBOOST_VERSION = 0.6
