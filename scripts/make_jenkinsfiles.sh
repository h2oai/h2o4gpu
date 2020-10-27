#!/bin/bash

# Make Jenkinsfiles.* to avoid copy/paste due to limitations of
# jenkins that stage names have to be static text labels

## declare an array variable
declare -a arr=("x86_64-cuda10-py36" "x86_64-cuda101-py36" "ppc64le-cuda10-py36" "ppc64le-cuda101-py36" "x86_64-cuda10-py37" "x86_64-cuda102-py38" "x86_64-cuda101-py37" "ppc64le-cuda10-py37" "ppc64le-cuda102-py38" "ppc64le-cuda101-py37")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   echo "#!/usr/bin/groovy" > ci/Jenkinsfile-$i
   echo "" >> ci/Jenkinsfile-$i
   echo "//################ FILE IS AUTO-GENERATED from .base files" >> ci/Jenkinsfile-$i
   echo "//################ DO NOT MODIFY" >> ci/Jenkinsfile-$i
   echo "//################ See scripts/make_jenkinsfiles.sh" >> ci/Jenkinsfile-$i
   echo "" >> ci/Jenkinsfile-$i

   cat ci/base/Jenkinsfile-$i.base >> ci/Jenkinsfile-$i
   echo "//################ BELOW IS COPY/PASTE of ci/Jenkinsfile.template (except stage names)" >> ci/Jenkinsfile-$i
   cat ci/Jenkinsfile.template >> ci/Jenkinsfile-$i

   sed -i.bck 's/stage\(.*\)\"/stage\1 '$i'\"/g' ci/Jenkinsfile-$i
   sed -i.bck 's/wasStageSuccessful\(.*\)\"/wasStageSuccessful\1 '$i'\"/g' ci/Jenkinsfile-$i

   if [[ $i == *"benchmark"* ]]; then
       echo "More for benchmarks"
       sed -i.bck 's/dobenchmark = \"1\"/dobenchmark = \"0\"/g' ci/Jenkinsfile-$i
       sed -i.bck 's/doruntime = \"1\"/doruntime = \"0\"/g' ci/Jenkinsfile-$i
   fi

   rm -rf ci/Jenkinsfile-$i.bck

done
