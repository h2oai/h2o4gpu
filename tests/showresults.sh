#!/usr/bin/env bash
path="results"
list=`ls $path | sort`
#for fil in $list ; do echo $fil ; done
rm -rf resultsall.dat
echo "Run Train CV Valid ..." >> resultsall.dat
for fil in $list ; do cat $path/$fil >> resultsall.dat ; done
column -t resultsall.dat
