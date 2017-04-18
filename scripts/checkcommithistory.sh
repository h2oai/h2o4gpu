export PATHORIG=$PATH
#export PATH=/usr/lib/R/bin:$PATHORIG
#myR=/usr/lib/R/bin/R
myR=R

pogsdir=~/pogs/
resultdir=~/results/
mkdir -p $resultdir
hashfullpath=~/hashes

cd ~/
rm -rf pogs
git clone git@github.com:h2oai/pogs.git
pushd $pogsdir
cp examples/R/simple.R $resultdir
git checkout master
git reset --hard origin/master
git log | grep "^commit\ [0-9]" | awk '{print $2}' > $hashfullpath
popd

rm -f $resultdir/resultsR.log
rm -f $resultdir/resultscpu.log
rm -f $resultdir/resultsgpu.log
for i in `cat $hashfullpath`; do
  echo "$i"
  cd $pogsdir
  git checkout $i
  #git reset --hard $i

  for j in `seq 0 1`; do
      cd $pogsdir/src/
      echo "Cleaning"
      make clean
      
      $myR CMD REMOVE pogs
      rm -rf pogs.a build/ *.dSYM
      find -name "*.so" -delete
      find -name "*.a" -delete
      find -name "*.o" -delete


      echo "Building: $PWD"
      cd interface_r/
      cd pogs
      ./cfg
      cd src
      if [ $j -eq 0 ]
      then
          sed -i 's/TARGET=gpu/TARGET=gpu/g' config.mk
          sed -i 's/TARGET=cpu/TARGET=gpu/g' config.mk
      else
          sed -i 's/TARGET=gpu/TARGET=cpu/g' config.mk
          sed -i 's/TARGET=cpu/TARGET=cpu/g' config.mk
      fi          
      ln -s ../../../gpu .
      ln -s ../../../cpu .
      ln -s ../../../include .
      #
      cd ..
      cd ..
      echo "Done: $PWD"
      rm -rf $HOME/R/x86_64-pc-linux-gnu-library/3.3/00LOCK-pogs
      MAKE="make -j32" $myR CMD INSTALL --build pogs


      
      rm -f pogs_1.0_R_x86_64-pc-linux-gnu.tar.gz
      git checkout pogs_1.0_R_x86_64-pc-linux-gnu.tar.gz
      rm -rf pogs/src/config.mk
      git checkout pogs/src/config.mk

      echo "Running $i $j"
      rm -rf $HOME/R/x86_64-pc-linux-gnu-library/3.3/00LOCK-pogs
      $myR -f $resultdir/simple.R | tee $resultdir/file.$i
      valpogs=`cat $resultdir/file.$i | grep '\[1\] \"RMSEPOGS' |awk '{print $2}'`
      itermax=`cat $resultdir/file.$i | grep " : " |awk '{print $1}'|grep -v Iter|grep -v TEST|grep -v Time|sort -n|tail -1`

      valglmstd=`cat $resultdir/file.$i | grep '\[1\] \"GLMNETSTDRMSE' |awk '{print $2}'`
      valglmnostd=`cat $resultdir/file.$i | grep '\[1\] \"GLMNETNOSTDRMSE' |awk '{print $2}'`
      valh2ostd=`cat $resultdir/file.$i | grep '\[1\] \"H2OSTDRMSE' |awk '{print $2}'`
      valh2onostd=`cat $resultdir/file.$i | grep '\[1\] \"H2ONOSTDRMSE' |awk '{print $2}'`

      echo $i $j $valpogs $itermax $valglmstd $valglmnostd $valh2ostd, $valh2onostd | tee -a $resultdir/resultsR.log
  done

  # C++
  cd $pogsdir/examples/cpp
  make -j all

  # get usage method
  #./runcpu
  usage=`./h2oai-glm-cpu &> use.txt ; cat use.txt | tail -1 | wc -w`
  if [ $usage -eq 8 ]
  then
      # then can do standardize on/off
      ./h2oai-glm-cpu 1 100 1 1 0 0 &> outcpu.txt
      ./h2oai-glm-cpu 1 100 1 1 1 0 &> outcpus.txt
      ./h2oai-glm-cpu-mapd 1 100 1 1 0 0 &> outcpu2.txt
      ./h2oai-glm-cpu-mapd 1 100 1 1 1 0 &> outcpus2.txt

      ./h2oai-glm-gpu 1 100 1 1 0 0 &> outgpu.txt
      ./h2oai-glm-gpu 1 100 1 1 1 0 &> outgpus.txt
      ./h2oai-glm-gpu-mapd 1 100 1 1 0 0 &> outgpu2.txt
      ./h2oai-glm-gpu-mapd 1 100 1 1 1 0 &> outgpus2.txt
  fi
  if [ $usage -eq 7 ]
  then
      # then no standardization
      ./h2oai-glm-cpu 1 100 1 1 0 &> outcpu.txt
      echo "NA" > outcpus.txt
      ./h2oai-glm-cpu-mapd 1 100 1 1 0 &> outcpu2.txt
      echo "NA" > outcpus2.txt
      ./h2oai-glm-gpu 1 100 1 1 0 &> outgpu.txt
      echo "NA" > outgpus.txt
      ./h2oai-glm-gpu-mapd 1 100 1 1 0 &> outgpu2.txt
      echo "NA" > outgpus2.txt
  fi
  if [ $usage -eq 6 ]
  then
      # then no intercept or standardization
      ./h2oai-glm-cpu 1 100 1 0 &> outcpu.txt
      echo "NA" > outcpus.txt
      ./h2oai-glm-cpu-mapd 1 100 1 0 &> outcpu2.txt
      echo "NA" > outcpus2.txt
      ./h2oai-glm-gpu 1 100 1 0 &> outgpu.txt
      echo "NA" > outgpus.txt
      ./h2oai-glm-gpu-mapd 1 100 1 0 &> outgpu2.txt
      echo "NA" > outgpus2.txt
  fi

  # just pick last, add |sort -rg before tail to sort and get best
  valcpu=`cat outcpu.txt |grep RMSE | awk '{print $14}' | tail -1`
  itercpu=`cat outcpu.txt | grep " : " |awk '{print $1}'|grep -v Iter|grep -v TEST|grep -v Time|sort -n|tail -1`
  if [[   -z  $valcpu  ]]
  then
      valcpu="NA      "
      itercpu="NA "
  fi
  valcpus=`cat outcpus.txt |grep RMSE | awk '{print $14}' | tail -1`
  itercpus=`cat outcpus.txt | grep " : " |awk '{print $1}'|grep -v Iter|grep -v TEST|grep -v Time|sort -n|tail -1`
  if [[   -z  $valcpus  ]]
  then
      valcpus="NA      "
      itercpus="NA "
  fi
  valcpu2=`cat outcpu2.txt |grep RMSE | awk '{print $14}' | tail -1`
  itercpu2=`cat outcpu2.txt | grep " : " |awk '{print $1}'|grep -v Iter|grep -v TEST|grep -v Time|sort -n|tail -1`
  if [[   -z  $valcpu2  ]]
  then
      valcpu2="NA      "
      itercpu2="NA "
  fi
  valcpus2=`cat outcpus2.txt |grep RMSE | awk '{print $14}' | tail -1`
  itercpus2=`cat outcpus2.txt | grep " : " |awk '{print $1}'|grep -v Iter|grep -v TEST|grep -v Time|sort -n|tail -1`
  if [[   -z  $valcpus2  ]]
  then
      valcpus2="NA      "
      itercpus2="NA "
  fi

  # GPU
  valgpu=`cat outgpu.txt |grep RMSE | awk '{print $14}' | tail -1`
  itergpu=`cat outgpu.txt | grep " : " |awk '{print $1}'|grep -v Iter|grep -v TEST|grep -v Time|sort -n|tail -1`
  if [[   -z  $valgpu  ]]
  then
      valgpu="NA      "
      itergpu="NA "
  fi
  valgpus=`cat outgpus.txt |grep RMSE | awk '{print $14}' | tail -1`
  itergpus=`cat outgpus.txt | grep " : " |awk '{print $1}'|grep -v Iter|grep -v TEST|grep -v Time|sort -n|tail -1`
  if [[   -z  $valgpus  ]]
  then
      valgpus="NA      "
      itergpus="NA "
  fi
  valgpu2=`cat outgpu2.txt |grep RMSE | awk '{print $14}' | tail -1`
  itergpu2=`cat outgpu2.txt | grep " : " |awk '{print $1}'|grep -v Iter|grep -v TEST|grep -v Time|sort -n|tail -1`
  if [[   -z  $valgpu2  ]]
  then
      valgpu2="NA      "
      itergpu2="NA "
  fi
  valgpus2=`cat outgpus2.txt |grep RMSE | awk '{print $14}' | tail -1`
  itergpus2=`cat outgpus2.txt | grep " : " |awk '{print $1}'|grep -v Iter|grep -v TEST|grep -v Time|sort -n|tail -1`
  if [[   -z  $valgpus2  ]]
  then
      valgpus2="NA      "
      itergpus2="NA "
  fi
  
  echo $i $j "R:"$valpogs $itermax "CPU10:"$valcpu $itercpu "CPU11:"$valcpus $itercpus "MAPD10:"$valcpu2 $itercpu2 "MAPD11:"$valcpus2 $itercpus2 | tee -a $resultdir/resultscpu.log
  echo $i $j "R:"$valpogs $itermax "GPU10:"$valgpu $itergpu "GPU11:"$valgpus $itergpus "MAPD10:"$valgpu2 $itergpu2 "MAPD11:"$valgpus2 $itergpus2 | tee -a $resultdir/resultsgpu.log
done
