#!/bin/bash

# Make Jenkinsfiles.* to avoid copy/paste due to limitations of
# jenkins that stage names have to be static text labels

## declare an array variable
declare -a arr=("nccl-cuda8" "nonccl-cuda9" "nccl-cuda9" "nonccl-cuda9" "nccl-cuda9-aws1" "nccl-cuda9-benchmark" "nccl-cuda9-aws1-benchmark")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   cat Jenkinsfile-$i.base >> Jenkinsfile-$i
   echo "//################ BELOW IS COPY/PASTE of Jenkinsfile.utils2 (except stage names)" >> Jenkinsfile-$i
   echo Jenkinsfile.utils2 >> Jenkinsfile-$i
   sed -i 's/stage\(.*\)\"/stage\1 $i\"/g' Jenkinsfile-$i

   if [[ $i == *"benchmark"* ]]; then
       echo "More for benchmarks"
       sed -i 's/dobenchmark = \"1\"/dobenchmark = \"0\"/g' Jenkinsfile-$i
       sed -i 's/doruntime = \"1\"/doruntime = \"0\"/g' Jenkinsfile-$i
   fi
   
done
