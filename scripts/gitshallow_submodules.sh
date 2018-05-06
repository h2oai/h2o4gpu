#!/bin/bash
git submodule init
for i in $(git submodule | awk '{print $2}'); do
    spath=$(git config -f .gitmodules --get submodule.$i.path)
    surl=$(git config -f .gitmodules --get submodule.$i.url)
    echo "submodule:" $i $spath $surl
    if [ $spath == "xgboost" ] ; then
        git submodule update $spath
    else
        git submodule update --depth 1 $spath
    fi
done
