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
rm -f $resultdir/results.log
for i in `cat $hashfullpath`; do
  echo "$i"
  pushd $pogsdir
  git checkout $i
  #git reset --hard $i
  cd src/
  echo "Cleaning"
  make clean

  $myR CMD REMOVE pogs
  rm -rf pogs.a build/ *.dSYM
  find -name "*.so" -delete
  find -name "*.a" -delete
  find -name "*.o" -delete

  echo "Building"
  echo $PWD
  cd interface_r/
  cd pogs
  ./cfg
  cd src
  sed -i 's/TARGET=gpu/TARGET=gpu/g' config.mk
  sed -i 's/TARGET=cpu/TARGET=gpu/g' config.mk
  ln -s ../../../gpu .
  ln -s ../../../cpu .
  ln -s ../../../include .
  #
  cd ..
  cd ..
  echo $PWD
  MAKE="make -j32" $myR CMD INSTALL --build pogs


  
  rm -f pogs_1.0_R_x86_64-pc-linux-gnu.tar.gz
  git checkout pogs_1.0_R_x86_64-pc-linux-gnu.tar.gz
  rm -rf pogs/src/config.mk
  git checkout pogs/src/config.mk

  popd
  echo "Running"
  $myR -f $resultdir/simple.R | tee $resultdir/file.$i
  val=`cat $resultdir/file.$i | grep '\[1\] \"RMSEPOGS' |awk '{print $2}'`
  itermax=`cat $resultdir/file.$i | grep " : " |awk '{print $1}'|grep -v Iter|grep -v TEST|grep -v Time|sort -n|tail -1`

  echo $i $val $itermax | tee -a $resultdir/results.log
done
