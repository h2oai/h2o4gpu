# save old file
mv src/interface_py/h2o4gpu/__init__.py src/interface_py/h2o4gpu/__init__.py.backup

# apply sklearn
rm -rf sklearn
cd scikit-learn
pip install dist/h2o4gpu-0.20.dev0-cp36-cp36m-linux_x86_64.whl --upgrade --target ../sklearn/
cd ../

# link-up recursively
sh ./scripts/importsklearn.sh

# handle base __init__.py file appending
rm -rf src/interface_py/h2o4gpu/__init__.py
mv src/interface_py/h2o4gpu/__init__.py.backup src/interface_py/h2o4gpu/__init__.py
cat sklearn/h2o4gpu/__init__.py | sed 's/__version__.*//g' >> src/interface_py/h2o4gpu/__init__.py

# register 
sh ./scripts/post_apply_sklearn.sh
