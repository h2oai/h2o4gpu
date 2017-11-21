#!/bin/bash

for fil in `ls examples/py/*.ipynb examples/py/demos/*.ipynb`
do
    base=`basename $fil`
    dir=`dirname $fil`
    name=$(echo $base | cut -f 1 -d '.')

    echo $fil $base $dir $name
    jupyter nbconvert --to script $fil
    # below for pytest
    mv $dir/$name.py $dir/test_$name.py
    # below for py.test
    cp $dir/$name.ipynb $dir/test_$name.ipynb

done
