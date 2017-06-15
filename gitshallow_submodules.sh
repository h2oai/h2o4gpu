
#!/bin/bash
git submodule init
for i in $(git submodule | sed -e 's/.* //'); do
    spath=$(git config -f .gitmodules --get submodule.$i.path)
    surl=$(git config -f .gitmodules --get submodule.$i.url)
    if [ $spath == "cub" ] || [ $spath == "nccl" ] ; then
        git submodule update --depth 1 $spath
    else
        git submodule update $spath
    fi
done
