
#!/bin/bash
git submodule init
for i in $(git submodule | awk '{print $2}'); do
    spath=$(git config -f .gitmodules --get submodule.$i.path)
    surl=$(git config -f .gitmodules --get submodule.$i.url)
    echo "submodule:" $i $spath $surl
#    if [ $spath == "cub" ] || [ $spath == "nccl" ] || [ $spath == "py3nvml" ] || [ $spath == "xgboost" ] ; then
    if [ $spath == "cub" ] || [ $spath == "nccl" ] || [ $spath == "py3nvml" ] ; then # can't add xgboost because not pulling from master
        git submodule update --depth 1 $spath
    else
        git submodule update $spath
    fi
done
