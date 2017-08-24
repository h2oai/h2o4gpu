#!/usr/bin/env bash
path="results"

############## RELATIVE ERRORS
list=`ls $path/*.error.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf resultsall.error.dat
echo "RunErrorRel Train CV Valid ..." >> resultsall.error.dat
for fil in $list ; do cat $fil >> resultsall.error.dat ; done
column -t resultsall.error.dat

echo "" 

############## H2O ERRORS
list=`ls $path/*.error.h2o.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf resultsall.error.h2o.dat
echo "RunErrorH2O Train CV Valid ..." >> resultsall.error.h2o.dat
for fil in $list ; do cat $fil >> resultsall.error.h2o.dat ; done
column -t resultsall.error.h2o.dat

echo "" 

############## H2O4GPU ERRORS
list=`ls $path/*.error.h2o4gpu.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf resultsall.error.h2o4gpu.dat
echo "RunErrorH2O4GPU Train CV Valid ..." >> resultsall.error.h2o4gpu.dat
for fil in $list ; do cat $fil >> resultsall.error.h2o4gpu.dat ; done
column -t resultsall.error.h2o4gpu.dat

echo "" 

############## RELATIVE TIMES
list=`ls $path/*.time.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf resultsall.time.dat
echo "RunTimeRatio Train CV Valid ..." >> resultsall.time.dat
for fil in $list ; do cat $fil >> resultsall.time.dat ; done
column -t resultsall.time.dat

echo "" 

############## H2O TIMES
list=`ls $path/*.time.h2o.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf resultsall.time.h2o.dat
echo "RunTimeH2O Train CV Valid ..." >> resultsall.time.h2o.dat
for fil in $list ; do cat $fil >> resultsall.time.h2o.dat ; done
column -t resultsall.time.h2o.dat

echo "" 

############## H2O4GPU TIMES
list=`ls $path/*.time.h2o4gpu.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf resultsall.time.h2o4gpu.dat
echo "RunTimeH2O4GPU Train CV Valid ..." >> resultsall.time.h2o4gpu.dat
for fil in $list ; do cat $fil >> resultsall.time.h2o4gpu.dat ; done
column -t resultsall.time.h2o4gpu.dat

echo "" 
