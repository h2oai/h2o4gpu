#!/bin/bash

# apply sklearn
bash ./scripts/apply_sklearn_pipinstall.sh

# link-up recursively
bash ./scripts/appkly_sklearn_link.sh

# handle base __init__.py file appending
bash ./scripts/apply_sklearn_initmerge.sh
