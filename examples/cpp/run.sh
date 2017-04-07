#!/bin/bash
mkdir -p ec2.100.16.0.2
git reflog | head -n1 | awk '{print $1}' > ec2.100.16.0.2/sha
make gpualt
make cpualt
for i in 16 8 4 2 1; do
   mkdir -p ec2.100.16.0.2/gpu.$i
   ./h2oai-glm-gpu $i 100 16 0.2 2>&1 | tee ec2.100.16.0.2/gpu.$i/pogs.ec2.100.16.0.2.gpu.$i.log
   mv me*txt ec2.100.16.0.2/gpu.$i/
done
mkdir -p ec2.100.16.0.2/cpu
./h2oai-glm-cpu 1 100 16 0.2 2>&1 | tee ec2.100.16.0.2/cpu/pogs.ec2.100.16.0.2.cpu.log
mv me*txt ec2.100.16.0.2/cpu/

