#!/bin/bash
L=100
A=16
V=0.2
DIR=$HOSTNAME.$L.$A.$V.`git reflog | head -n1 | awk '{print $1}'`

rm -f me*.txt
mkdir -p $DIR
make gpualt
make cpualt
for i in 2 1; do
   mkdir -p $DIR/gpu.$i
   ./h2o4gpu-glm-gpu $i $L $A $V 2>&1 | tee $DIR/gpu.$i/pogs.$DIR.gpu.$i.log
   mv me*txt $DIR/gpu.$i/
done
mkdir -p $DIR/cpu
./h2o4gpu-glm-cpu 1 $L $A $V 2>&1 | tee $DIR/cpu/pogs.$DIR.cpu.log
mv me*txt $DIR/cpu/

