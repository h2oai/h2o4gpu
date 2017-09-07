#!/usr/bin/env bash
path="results"

############## RELATIVE ERRORS with masks
list=`ls $path/*.error.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf $path/results.error.all.dat
echo "RunErrorRel Train CV Valid ..." >> $path/results.error.all.dat
for fil in $list ; do cat $fil >> $path/results.error.all.dat ; done
column -t $path/results.error.all.dat

echo "" 

############## RELATIVE ERRORS
list=`ls $path/*.error.quick.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf $path/results.error.quick.all.dat
echo "RunErrorRel Train CV Valid ..." >> $path/results.error.quick.all.dat
for fil in $list ; do cat $fil >> $path/results.error.quick.all.dat ; done
column -t $path/results.error.quick.all.dat

echo "" 

############## H2O ERRORS
list=`ls $path/*.error.h2o.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf $path/results.error.h2o.all.dat
echo "RunErrorH2O Train CV Valid ..." >> $path/results.error.h2o.all.dat
for fil in $list ; do cat $fil >> $path/results.error.h2o.all.dat ; done
column -t $path/results.error.h2o.all.dat

echo "" 

############## H2O4GPU ERRORS
list=`ls $path/*.error.h2o4gpu.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf $path/results.error.h2o4gpu.all.dat
echo "RunErrorH2O4GPU Train CV Valid ..." >> $path/results.error.h2o4gpu.all.dat
for fil in $list ; do cat $fil >> $path/results.error.h2o4gpu.all.dat ; done
column -t $path/results.error.h2o4gpu.all.dat

echo "" 

############## RELATIVE TIMES
list=`ls $path/*.time.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf $path/results.time.all.dat
echo "RunTimeRatio Time_Ratio" >> $path/results.time.all.dat
for fil in $list ; do cat $fil >> $path/results.time.all.dat ; done
column -t $path/results.time.all.dat

echo "" 

############## H2O TIMES
list=`ls $path/*.time.h2o.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf $path/results.time.h2o.all.dat
echo "RunTimeH2O Time_H2O" >> $path/results.time.h2o.all.dat
for fil in $list ; do cat $fil >> $path/results.time.h2o.all.dat ; done
column -t $path/results.time.h2o.all.dat

echo "" 

############## H2O4GPU TIMES
list=`ls $path/*.time.h2o4gpu.dat | sort`
#for fil in $list ; do echo $fil ; done
rm -rf $path/results.time.h2o4gpu.all.dat
echo "RunTimeH2O4GPU Time_H2O4GPU" >> $path/results.time.h2o4gpu.all.dat
for fil in $list ; do cat $fil >> $path/results.time.h2o4gpu.all.dat ; done
column -t $path/results.time.h2o4gpu.all.dat

echo "" 
