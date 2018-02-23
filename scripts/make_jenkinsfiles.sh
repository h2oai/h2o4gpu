#!/bin/bash

# Make Jenkinsfiles.* to avoid copy/paste due to limitations of
# jenkins that stage names have to be static text labels

## declare an array variable
declare -a arr=("nccl-cuda8" "nonccl-cuda9" "nccl-cuda9" "cpu" "nonccl-cuda9" "nccl-cuda9-aws1" "nccl-cuda9-benchmark" "nccl-cuda9-aws1-benchmark" "cpu" "nonccl-cuda8-centos")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   echo "#!/usr/bin/groovy" > Jenkinsfile-$i
   echo "" >> Jenkinsfile-$i
   echo "//################ FILE IS AUTO-GENERATED from .base files" >> Jenkinsfile-$i
   echo "//################ DO NOT MODIFY" >> Jenkinsfile-$i
   echo "//################ See scripts/make_jenkinsfiles.sh" >> Jenkinsfile-$i
   echo "" >> Jenkinsfile-$i

   cat Jenkinsfile-$i.base >> Jenkinsfile-$i
   echo "//################ BELOW IS COPY/PASTE of Jenkinsfile.utils2 (except stage names)" >> Jenkinsfile-$i
   cat Jenkinsfile.utils2 >> Jenkinsfile-$i
   sed -i 's/stage\(.*\)\"/stage\1 '$i'\"/g' Jenkinsfile-$i

   if [[ $i == *"benchmark"* ]]; then
       echo "More for benchmarks"
       sed -i 's/dobenchmark = \"1\"/dobenchmark = \"0\"/g' Jenkinsfile-$i
       sed -i 's/doruntime = \"1\"/doruntime = \"0\"/g' Jenkinsfile-$i
   fi
   
done
