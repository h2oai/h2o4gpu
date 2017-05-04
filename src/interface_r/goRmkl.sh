#new 2016, but flawed: https://software.intel.com/en-us/articles/build-r-301-with-intel-c-compiler-and-intel-mkl-on-linux
#old 2014: https://software.intel.com/en-us/articles/using-intel-mkl-with-r
#old 2014: https://software.intel.com/en-us/articles/quick-linking-intel-mkl-blas-lapack-to-r
#old 2012: https://www.r-bloggers.com/speeding-up-r-with-intels-math-kernel-library-mkl/
#see also 2016: https://support.rstudio.com/hc/en-us/articles/218004217-Building-R-from-source
#see for using any blas with R, including cuda: https://www.r-bloggers.com/r-benchmark-for-high-performance-analytics-and-computing-i/



#To get Ubuntu 16.04 oracle java 1.8: https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04

#Working instructions, do:

source /opt/intel/bin/compilervars.sh intel64
export JAVA_HOME=/usr/lib/jvm/java-8-oracle/

# e.g.
wget https://cran.r-project.org/src/base/R-3/R-3.3.3.tar.gz
# see as alternative: https://mran.revolutionanalytics.com/download/
tar xvzf R-3.3.3.tar.gz
cd R-3.3.3/
# note mkl_gf_lp64 instead of mkl_intel_lp64
CC="icc" CXX="icpc" F77="ifort" FC="ifort" AR="xiar" LD="xild" CFLAGS="-O3 -openmp -xHost" CXXFLAGS="-O3  -openmp -xHost" F77FLAGS="-O3 -openmp -xHost" FFLAGS="-O3 -openmp -xHost" MKL="-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread" ./configure --with-blas="$MKL" --with-lapack &> configure.log

make &> make.log

sudo bash
source /opt/intel/bin/compilervars.sh intel64
CC="icc" CXX="icpc" F77="ifort" FC="ifort" AR="xiar" LD="xild" CFLAGS="-O3 -openmp -xHost" CXXFLAGS="-O3  -openmp -xHost" F77FLAGS="-O3 -openmp -xHost" FFLAGS="-O3 -openmp -xHost" MKL="-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread"
make install
# installs in /usr/local

# test performance (still in R-3.3.3 directory):
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
# get code from first window at: https://software.intel.com/en-us/articles/extending-r-with-intel-mkl and call it pow_wrp.c
$CC $CFLAGS -fPIC -Iinclude -c pow_wrp.c -o pow_wrp.o
$CC $CFLAGS -shared -openmp -liomp5 -lmkl_rt -o pow_wrp.so pow_wrp.o -L./lib -lR
# run on dual Intel(R) Xeon(R) CPU E5-2687W 0 @ 3.10GHz
# edit pow.R to have 100X as many n
./bin/Rscript pow.R
# first result is special vectorized MKL power, 2nd is loop only available otherwise (a beyond blas result)
#   user  system elapsed 
# 20.604   0.668   1.377 
#   user  system elapsed 
#173.528   0.048 172.976
# note: only uses 100% cpu with 32 hyperthreaded cores, according to "top"

#####################################
# check general performance
#http://r.research.att.com/benchmarks/
#https://www.r-bloggers.com/r-benchmark-for-high-performance-analytics-and-computing-i/
#http://edustatistics.org/nathanvan/category/rstats/
export  MKL_INTERFACE_LAYER=GNU,LP64
export  MKL_THREADING_LAYER=GNU

wget http://r.research.att.com/benchmarks/R-benchmark-25.R
#Add "library('methods') and library('utils') to top of R-benchmark-25.R
#see: https://github.com/dhimmel/elevcan/issues/1
./bin/Rscript R-benchmark-25.R
# result:
#Total time for all 15 tests_________________________ (sec):  14.3753333333333 
#Overall mean (sum of I, II and III trimmed means/3)_ (sec):  0.527919893914026
# note: only a few tests actually fully use OpenMP (maxing out multiple cores), like the "Inverse of a 1600x1600 random matrix" at near max of 3200% cpu usage in R.  Not sure why, probably because blas only used for some things but not all and mkl can't help unless written for blas/lapack.

# with normal Ubuntu install (optimized openblas):
/usr/lib/R/bin/Rscript R-benchmark-25.R
#Total time for all 15 tests_________________________ (sec):  12.479 
#Overall mean (sum of I, II and III trimmed means/3)_ (sec):  0.461772914460023
# more often uses more than 1 core

# with microsoft:
/usr/lib64/microsoft-r/3.3/lib64/R/bin/Rscript R-benchmark-25.R
#Total time for all 15 tests_________________________ (sec):  14.555 
#Overall mean (sum of I, II and III trimmed means/3)_ (sec):  0.493125171209649





#################### POGS
git clone git@github.com:h2oai/h2oaiglm.git
cd h2oaiglm
cd src
emacs -nw Makefile
# and switch to:
# below doesn't work:
#CXX=icc
#CXXFLAGS=$(IFLAGS) -g -O3 -Wall -openmp -xHost
# below works:
CXX=icpc
CXXFLAGS=$(IFLAGS) -g -O3 -Wall -std=c++11 -fPIC -openmp -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread
make cpu

cd interface_r/h2oaiglm
./cfg
emacs -nw src/config.mk # and edit TARGET as cpu (for MKL blas), edit RHOME binary and lib path to point to new R (if in R directory at this point, R RHOME should be correctly new R)

emacs -nw src/Makefile # and change CXXFLAGS as above

cd ../
R CMD INSTALL --build h2oaiglm

R
install.packages("h2oaiglm.0_R_x86_64-pc-linux-gnu.tar.gz", repos = NULL, type="source")
# installs in ~/R/x86_64-pc-linux-gnu-library/3.3/h2oaiglm/ (assumes overwrites system's libraries)

cd ../interface_c
emacs Makefile # edit CXXFLAGS as above

cd ../../examples/c
emacs -nw Makefile # change -lopenblas to -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
make cpu
time ./run
# result:
#START
#----------------------------------------------------------------------------
# Iter | pri res | pri tol | dua res | dua tol |   gap   | eps gap | pri obj
#----------------------------------------------------------------------------
#    0 : 2.35e+00  1.07e-02  2.34e+00  3.16e-02  5.50e+00  3.82e-02  8.70e-03
#----------------------------------------------------------------------------
#Status: Reached max iter
#Timing: Total = 3.99e+01 s, Init = 2.82e+01 s
#Iter  : 49
#----------------------------------------------------------------------------
#Error Metrics:
#Pri: |Ax - y|    / (abs_tol sqrt(m)     / rel_tol + |y|)          = 1.15e-01
#Dua: |A'l + u|   / (abs_tol sqrt(n)     / rel_tol + |u|)          = 1.00e-03
#Gap: |x'u + y'l| / (abs_tol sqrt(m + n) / rel_tol + |x,u| |y,l|)  = 3.80e-01
#----------------------------------------------------------------------------
#Lasso optval = 2.592653e+02, final iter = 49
#
#real    0m47.284s
#user    7m33.520s
#sys     0m2.868s

# with original non-MKL cpu run:
time ./runnew
#START
#----------------------------------------------------------------------------
# Iter | pri res | pri tol | dua res | dua tol |   gap   | eps gap | pri obj
#----------------------------------------------------------------------------
#    0 : 2.35e+00  1.07e-02  2.34e+00  3.16e-02  5.50e+00  3.82e-02  8.70e-03
#----------------------------------------------------------------------------
#Status: Reached max iter
#Timing: Total = 5.19e+01 s, Init = 3.41e+01 s
#Iter  : 49
#----------------------------------------------------------------------------
#Error Metrics:
#Pri: |Ax - y|    / (abs_tol sqrt(m)     / rel_tol + |y|)          = 1.15e-01
#Dua: |A'l + u|   / (abs_tol sqrt(n)     / rel_tol + |u|)          = 1.00e-03
#Gap: |x'u + y'l| / (abs_tol sqrt(m + n) / rel_tol + |x,u| |y,l|)  = 3.80e-01
#----------------------------------------------------------------------------
#Lasso optval = 2.592605e+02, final iter = 49
#
#real    1m5.357s
#user    13m52.176s
#sys     5m53.488s

# with original gpu run:
time ./run
#START
#----------------------------------------------------------------------------
# Iter | pri res | pri tol | dua res | dua tol |   gap   | eps gap | pri obj
#----------------------------------------------------------------------------
#    0 : 2.35e+00  1.07e-02  2.34e+00  3.16e-02  5.50e+00  3.82e-02  8.70e-03
#----------------------------------------------------------------------------
#Status: Reached max iter
#Timing: Total = 6.87e+00 s, Init = 4.52e+00 s
#Iter  : 49
#----------------------------------------------------------------------------
#Error Metrics:
#Pri: |Ax - y|    / (abs_tol sqrt(m)     / rel_tol + |y|)          = 5.86e-02
#Dua: |A'l + u|   / (abs_tol sqrt(n)     / rel_tol + |u|)          = 3.23e-03
#Gap: |x'u + y'l| / (abs_tol sqrt(m + n) / rel_tol + |x,u| |y,l|)  = 1.24e-01
#----------------------------------------------------------------------------
#Lasso optval = 5.652949e+02, final iter = 49
#
#real    0m16.694s
#user    0m14.188s
#sys     0m2.488s

# issue is not using more than 1 core, often doesn't with any test.  Sometimes does. Not sure why.


export MKL_NUM_THREADS=32
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=false
unset MKL_INTERFACE_LAYER
unset MKL_THREADING_LAYER

###

# see as alternative: https://mran.revolutionanalytics.com/download/
#then do
/usr/lib64/microsoft-r/3.3/lib64/R/bin/R
/usr/local/bin/R
/usr/lib/R/bin/R
