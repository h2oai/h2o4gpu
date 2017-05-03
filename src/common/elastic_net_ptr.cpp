#include "elastic_net_ptr.h"
#include <float.h>
#include "../include/util.h"

#if(USEMKL==1)
#include <mkl.h>
#endif


#ifdef HAVECUDA
#define TEXTARCH "GPU"

#define TEXTBLAS "CUDA"

#else

#define TEXTARCH "CPU"

#if(USEMKL==1)
#define TEXTBLAS "MKL"
#else
#define TEXTBLAS "CPU"
#endif

#endif

#if(USEICC==1)
#define TEXTCOMP "ICC"
#else
#define TEXTCOMP "GCC"
#endif

using namespace std;

string cmd(const string& cmd) {
  FILE *fp;
  char c[1025];
  string res;

  /* Open the command for reading. */
  fp = popen(cmd.c_str(), "r");
  if (fp == NULL) {
    printf("Failed to run command\n" );
    return "NA";
  }
  fgets(c, sizeof(c)-1, fp); //only take the first line
  res += c;
  pclose(fp);
  res.erase(std::remove(res.begin(), res.end(), '\n'),res.end());
  //  cout << "cmd: " << res << endl;
  return res;
}

const std::string CPUTYPE = cmd("lscpu | grep 'Model name' | cut -d: -f2- | sed 's/ \\+//g' | sed 's/Intel(R)//' | sed 's/Core(TM)//' | sed 's/CPU//'");
const std::string SOCKETS = cmd("lscpu | grep 'Socket(s)' | cut -d: -f2- | sed 's/ \\+//g'");

const std::string GPUTYPE = cmd("nvidia-smi -q | grep 'Product Name' | cut -d: -f2- | sed 's/ \\+//g' | tail -n 1");
const std::string NGPUS = cmd("nvidia-smi -q | grep 'Product Name' | cut -d: -f2- | sed 's/ \\+//g' | wc -l");

#ifdef HAVECUDA
const std::string HARDWARE = NGPUS + "x" + GPUTYPE;
#else
const std::string HARDWARE = SOCKETS + "x" + CPUTYPE;
#endif


#define Printmescore(thefile)  fprintf(thefile,                         \
                                       "%s.me: %d ARCH: %s:%s BLAS: %s%d COMP: %s sharedA: %d nThreads: %d nGPUs: %d time: %21.15g lambdatype: %d fi: %d a: %d alpha: %g intercept: %d standardize: %d i: %d " \
                                       "lambda: %g dof: %d trainRMSE: %f ivalidRMSE: %f validRMSE: %f\n", \
                                       _GITHASH_, me, TEXTARCH, HARDWARE.c_str(), TEXTBLAS, blasnumber, TEXTCOMP, sharedA, nThreads, nGPUs, timer<double>(), lambdatype, fi, a, alpha,intercept,standardize, (int)i, \
                                       lambda, (int)dof, trainRMSE, ivalidRMSE, validRMSE); fflush(thefile);

#define Printmescoresimple(thefile)  fprintf(thefile,"%21.15g %d %d %21.15g %21.15g %15.7f %15.7f %15.7f\n", timer<double>(), lambdatype, fi, alpha, lambda, trainRMSE, ivalidRMSE, validRMSE); fflush(thefile);

#define PrintmescoresimpleCV(thefile,lambdatype,bestalpha,bestlambda,bestrmse1,bestrmse2,bestrmse3)  fprintf(thefile,"BEST: %21.15g %d %21.15g %21.15g %15.7f %15.7f %15.7f\n", timer<double>(), lambdatype, bestalpha, bestlambda, bestrmse1,bestrmse2,bestrmse3 ); fflush(thefile);


#include <stdio.h>
#include <stdlib.h>
#include <signal.h> //  our new library

#define OLDPRED 0 // JONTODO: cleanup: if OLDPRED=1, then must set sharedAlocal=0 in examples/cpp/elastic_net_ptr_driver.cpp when doing make pointer part, so that don't overwrite original data (due to equilibration) so can be used for scoring.

#define DOSTOPEARLY 1

namespace h2oaiglm {

  volatile sig_atomic_t flag = 0;
  inline void my_function(int sig){ // can be called asynchronously
    fprintf(stderr, "Caught signal %d. Terminating shortly.\n", sig);
    flag = 1; // set flag
  }

  bool stopEarly(vector<double> val, int k, double tolerance, bool moreIsBetter, bool verbose, double norm, double *jump) {
    if (val.size()-1 < 2*k) return false; //need 2k scoring events (+1 to skip the very first one, which might be full of NaNs)
    vector<double> moving_avg(k+1); //one moving avg for the last k+1 scoring events (1 is reference, k consecutive attempts to improve)

    // compute moving average(s)
    for (int i=0;i<moving_avg.size();++i) {
      moving_avg[i]=0;
      int startidx=val.size()-2*k+i;
      for (int j=0;j<k;++j)
        moving_avg[i]+=val[startidx+j];
      moving_avg[i]/=k;
    }
    if (verbose) {
      cout << "JUnit: moving averages: ";
      copy(moving_avg.begin(), moving_avg.end(), ostream_iterator<double>(cout, " "));
      cout << endl;
    }

    // get average of moving average
    double moving_avgavg=0;
    {
      int i;
      for (i=0;i<moving_avg.size();++i) {
        moving_avgavg+=moving_avg[i];
      }
      moving_avgavg/=((double)i);
    }

    // var variance and rmse of moving average
    double var=0,rmse=0;
    {
      int i;
      for (i=0;i<moving_avg.size();++i) {
        var += pow(moving_avg[i]-moving_avgavg,2.0);
      }
      rmse=sqrt(var)/((double)i);
    }

    // check if any of the moving averages is better than the reference (by at least tolerance relative improvement)
    double ref = moving_avg[0];
    bool improved = false;
    for (int i=1;i<moving_avg.size();++i) {
      //      fprintf(stderr,"ref=%g tol=%g moving=%g i=%d moving_avgavg=%g rmse=%g\n",ref,tolerance,moving_avg[i],i,moving_avgavg,rmse); fflush(stderr);
      if (moreIsBetter)
        improved |= (moving_avg[i] > ref*(1.0+tolerance));
      else
        improved |= (moving_avg[i] < ref*(1.0-tolerance));
    }

    // estimate normalized jump for controlling tolerance as approach stopping point
    if(moving_avg.size()>=2){
      //      *jump = (*std::max_element(moving_avg.begin(), moving_avg.end()) - *std::min_element(moving_avg.begin(), moving_avg.end()))/(DBL_EPSILON+norm);
      *jump = (moving_avg.front() - moving_avg.back())/(DBL_EPSILON+ moving_avg.front()+ moving_avg.back());
    }
    else{
      *jump=DBL_MAX;
    }

      
    if (improved) {
      if (improved && verbose)
        cout << "improved from " << ref << " to " << (moreIsBetter ? *std::max_element(moving_avg.begin(), moving_avg.end()) : *std::min_element(moving_avg.begin(), moving_avg.end())) << endl;
      return false;
    }
    else {
      if (verbose) cout << "stopped." << endl;
      return true;
    }
  }







  // Elastic Net
  //   minimize    (1/2) ||Ax - b||_2^2 + \lambda \alpha ||x||_1 + \lambda 1-\alpha ||x||_2
  //
  // for many values of \lambda and multiple values of \alpha
  // See <h2oaiglm>/matlab/examples/lasso_path.m for detailed description.
  // m and n are training data size
#define NUMRMSE 3 // train, hold-out CV, valid
#define NUMOTHER 3 // for lambda, alpha, tol
  template<typename T>
  double ElasticNetptr(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                       size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                       double lambda_min_ratio, int nLambdas, int nFolds, int nAlphas,
                       void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr
                       ,int givefullpath
                       ,T **Xvsalphalambda, T **Xvsalpha
                       ) {

    signal(SIGINT, my_function);
    signal(SIGTERM, my_function);
    int nlambda = nLambdas;
    if (nlambda <= 1) {
      cerr << "Must use nlambda > 1\n";
      exit(-1);
    }

    cout << "Hardware: " << HARDWARE << endl;

    // number of openmp threads = number of cuda devices to use
#ifdef _OPENMP
    int omt=omp_get_max_threads();
    //      omp_set_num_threads(MIN(omt,nGPUs));  // not necessary, but most useful mode so far
    omp_set_num_threads(nThreads);  // not necessary, but most useful mode so far
    int nth=omp_get_max_threads();
    //      nGPUs=nth; // openmp threads = cuda/cpu devices used
    omp_set_dynamic(0);
#if(USEMKL==1)
    mkl_set_dynamic(0);
#endif
    omp_set_nested(1);
    omp_set_max_active_levels(2);
#ifdef DEBUG
    cout << "Number of original threads=" << omt << " Number of final threads=" << nth << endl;
#endif
    if (nAlphas % nThreads != 0) {
      DEBUG_FPRINTF(stderr, "NOTE: Number of alpha's not evenly divisible by number of Threads, so not efficint load balancing: %d\n",0);
    }
#endif

    
    // report fold setup
    size_t totalfolds=nFolds*( nFolds>1 ? 2 : 1 );
    fprintf(stderr,"Real folds=%d Total Folds=%zu\n",nFolds,totalfolds); fflush(stderr);


    // setup storage for returning results back to user
    // iterate over predictors (n) or other information fastest so can memcpy X
#define MAPXALL(i,a,which) (which + a*(n+NUMRMSE+NUMOTHER) + i*(n+NUMRMSE+NUMOTHER)*nLambdas)
#define MAPXBEST(a,which) (which + a*(n+NUMRMSE+NUMOTHER))
    if(givefullpath){
      *Xvsalphalambda = (T*) calloc(sizeof(T)*nLambas*nAlphas*(n+NUMRMSE+NUMOTHER)); // +3 for values of lambda, alpha, and tolerance
    }
    else{ // only give back solution for optimal lambda after CV is done
      *Xvsalphalambda = NULL;
    }
    *Xvsalpha = (T*) calloc(sizeof(T)*nAlphas*(n+NUMRMSE+NUMOTHER));
    

    // for source, create class objects that creates cuda memory, cpu memory, etc.
    // This takes-in raw GPU pointer
    //  h2oaiglm::MatrixDense<T> Asource_(sourceDev, ord, mTrain, n, mValid, reinterpret_cast<T *>(trainXptr));
    // assume source thread is 0th thread (TODO: need to ensure?)
    int sourceme=sourceDev;
    h2oaiglm::MatrixDense<T> Asource_(sharedA, sourceme, sourceDev, datatype, ord, mTrain, n, mValid,
                                      reinterpret_cast<T *>(trainXptr), reinterpret_cast<T *>(trainYptr),
                                      reinterpret_cast<T *>(validXptr), reinterpret_cast<T *>(validYptr),
                                      reinterpret_cast<T *>(weightptr));
    // now can always access A_(sourceDev) to get pointer from within other MatrixDense calls
    T min[2], max[2], mean[2], var[2], sd[2], skew[2], kurt[2];
    T lambdamax0;
    Asource_.Stats(intercept,min,max,mean,var,sd,skew,kurt,lambdamax0);
    double sdTrainY=(double)sd[0], meanTrainY=(double)mean[0];
    double sdValidY=(double)sd[1], meanValidY=(double)mean[1];
    double lambda_max0 = (double)lambdamax0;

    fprintf(stderr,"min %21.15g %21.15g\n",min[0],min[1]);
    fprintf(stderr,"max %21.15g %21.15g\n",max[0],max[1]);
    fprintf(stderr,"mean %21.15g %21.15g\n",mean[0],mean[1]);
    fprintf(stderr,"var %21.15g %21.15g\n",var[0],var[1]);
    fprintf(stderr,"sd %21.15g %21.15g\n",sd[0],sd[1]);
    fprintf(stderr,"skew %21.15g %21.15g\n",skew[0],skew[1]);
    fprintf(stderr,"kurt %21.15g %21.15g\n",kurt[0],kurt[1]);
    cout << "lambda_max0 " << lambda_max0 << endl;
    //      exit(0);
      
    // temporarily get trainX, etc. from h2oaiglm (which may be on gpu)
    T *trainX=NULL;
    T *trainY=NULL;
    T *validX=NULL;
    T *validY=NULL;
    T *trainW=NULL;
    if(OLDPRED) trainX = (T *) malloc(sizeof(T) * mTrain * n);
    trainY = (T *) malloc(sizeof(T) * mTrain);
    if(OLDPRED) validX = (T *) malloc(sizeof(T) * mValid * n);
    validY = (T *) malloc(sizeof(T) * mValid);
    trainW = (T *) malloc(sizeof(T) * mTrain);

    if(OLDPRED) Asource_.GetTrainX(datatype, mTrain * n, &trainX);
    Asource_.GetTrainY(datatype, mTrain, &trainY);
    if(OLDPRED) Asource_.GetValidX(datatype, mValid * n, &validX);
    Asource_.GetValidY(datatype, mValid, &validY);
    Asource_.GetWeight(datatype, mTrain, &trainW);


    T alphaarray[nFolds*2][nAlphas]; // shared memory space for storing alpha for various folds and alphas
    T lambdaarray[nFolds*2][nAlphas]; // shared memory space for storing lambda for various folds and alphas
    T tolarray[nFolds*2][nAlphas]; // shared memory space for storing tolerance for various folds and alphas
    // which rmse to use for final check of which model is best (keep validation fractional data for purely reporting)
    int owhichrmse;
    if(mValid>0){
      if(nFolds<=1) owhichrmse=2;
      else owhichrmse=2;
    }
    else{
      if(nFolds<=1) owhichrmse=0;
      else owhichrmse=1;
    }
    // which rmse to use within lambda-loop to decide if accurate model
    int iwhichrmse;
    if(mValid>0){
      if(nFolds<=1) iwhichrmse=2;
      else iwhichrmse=1;
    }
    else{
      if(nFolds<=1) iwhichrmse=0;
      else iwhichrmse=1;
    }
#define RMSELOOP(ri) for(int ri=0;ri<NUMRMSE;ri++)
    T rmsearray[NUMRMSE][nFolds*2][nAlphas]; // shared memory space for storing rmse for various folds and alphas
#define MAX(a,b) ((a)>(b) ? (a) : (b))
    // Setup each thread's h2oaiglm
    double t = timer<double>();
    double t1me0;


    ////////////////////////////////
    // PARALLEL REGION
#pragma omp parallel proc_bind(master)
    {
#ifdef _OPENMP
      int me = omp_get_thread_num();
      //https://software.intel.com/en-us/node/522115
      int physicalcores=omt;///2; // asssume hyperthreading Intel processor (doens't improve much to ensure physical cores used0
      // set number of mkl threads per openmp thread so that not oversubscribing cores
      int mklperthread=MAX(1,(physicalcores % nThreads==0 ? physicalcores/nThreads : physicalcores/nThreads+1));
#if(USEMKL==1)
      //mkl_set_num_threads_local(mklperthread);
      mkl_set_num_threads_local(mklperthread);
      //But see (hyperthreading threads not good for MKL): https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/288645
#endif
#else
      int me=0;
#endif

      int blasnumber;
#ifdef HAVECUDA
      blasnumber=CUDA_MAJOR;
#else
      blasnumber=mklperthread; // roughly accurate for openblas as well
#endif
        
      // choose GPU device ID for each thread
      int wDev = (nGPUs>0 ? me%nGPUs : 0);

      // Setup file output
      char filename[100];
      sprintf(filename, "me%d.%d.%s.%s.%d.%d.%d.txt", me, wDev, _GITHASH_, TEXTARCH, sharedA, nThreads, nGPUs);
      FILE *fil = fopen(filename, "wt");
      if (fil == NULL) {
        cerr << "Cannot open filename=" << filename << endl;
        exit(0);
      }
      else fflush(fil);
      sprintf(filename, "me%d.latest.txt", me); // for each thread
      FILE *fillatest = fopen(filename, "wt");
      if (fillatest == NULL) {
        cerr << "Cannot open filename=" << filename << endl;
        exit(0);
      }

      ////////////
      //
      // create class objects that creates cuda memory, cpu memory, etc.
      //
      ////////////
      double t0 = timer<double>();
      DEBUG_FPRINTF(fil, "Moving data to the GPU. Starting at %21.15g\n", t0);
#pragma omp barrier // not required barrier
      h2oaiglm::MatrixDense<T> A_(sharedA, me, wDev, Asource_);
#pragma omp barrier // required barrier for wDev=sourceDev so that Asource_._data (etc.) is not overwritten inside h2oaiglm_data(wDev=sourceDev) below before other cores copy data
      h2oaiglm::H2OAIGLMDirect<T, h2oaiglm::MatrixDense<T> > h2oaiglm_data(sharedA, me, wDev, A_);
#pragma omp barrier // not required barrier
      double t1 = timer<double>();
      if(me==0){ //only thread=0 times entire post-warmup procedure
        t1me0=t1;
      }
      DEBUG_FPRINTF(fil, "Done moving data to the GPU. Stopping at %21.15g\n", t1);
      DEBUG_FPRINTF(fil, "Done moving data to the GPU. Took %g secs\n", t1 - t0);

      // Setup constant parameters for all models
      h2oaiglm_data.SetnDev(1); // set how many cuda devices to use internally in h2oaiglm
      //    h2oaiglm_data.SetRelTol(1e-4); // set how many cuda devices to use internally in h2oaiglm
      //    h2oaiglm_data.SetAbsTol(1e-5); // set how many cuda devices to use internally in h2oaiglm
      //    h2oaiglm_data.SetAdaptiveRho(true);
      //h2oaiglm_data.SetEquil(false);
      //    h2oaiglm_data.SetRho(1);
      //	h2oaiglm_data.SetVerbose(5);
      //        h2oaiglm_data.SetMaxIter(100);

      DEBUG_FPRINTF(fil, "BEGIN SOLVE: %d\n",0);
      int fi,a;


      T *X0 = new T[n]();
      T *L0 = new T[mTrain]();
      int gotpreviousX0=0;


      ////////////////////////////
      //
      // loop over normal lambda path and then final cross folded lambda model
      //
      ////////////////////////////
#define LAMBDATYPEPATH 0
#define LAMBDATYPEONE 1
      // store various model parameters as averaged over folds during lambda-path, so that when do one final model with fixed lambda have model.
      double alphaarrayofa[nAlphas];
      double lambdaarrayofa[nAlphas];
      double tolarrayofa[nAlphas];
      double rmsearrayofa[NUMRMSE][nAlphas];
      for(int lambdatype=0;lambdatype<=(nFolds>1);lambdatype++){
        size_t nlambdalocal;


        // Set Lambda
        std::vector <T> lambdas(nlambda);
        if(lambdatype==LAMBDATYPEPATH){
          nlambdalocal=nlambda;
          const T lambda_min = lambda_min_ratio * static_cast<T>(lambda_max0); // like h2oaiglm.R
          T lambda_max = lambda_max0; // std::max(static_cast<T>(1e-2), alpha); // same as H2O
          DEBUG_FPRINTF(stderr, "lambda_max: %f\n", lambda_max);
          DEBUG_FPRINTF(stderr, "lambda_min: %f\n", lambda_min);
          DEBUG_FPRINTF(fil, "lambda_max: %f\n", lambda_max);
          DEBUG_FPRINTF(fil, "lambda_min: %f\n", lambda_min);
          // Regularization path: geometric series from lambda_max to lambda_min
          double dec = std::pow(lambda_min_ratio, 1.0 / (nlambdalocal - 1.));
          lambdas[0] = lambda_max;
          for (int i = 1; i < nlambdalocal; ++i)
            lambdas[i] = lambdas[i - 1] * dec;
        }
        else{
          nlambdalocal=1;
        }


        
        //////////////////////////////
        //
        // LOOP OVER FOLDS AND ALPHAS
        //
        ///////////////////////////////
#pragma omp for schedule(dynamic,1) collapse(2)
        for (a = 0; a < nAlphas; ++a) { //alpha search
          for (fi = 0; fi < nFolds; ++fi) { //fold

            ////////////
            // SETUP ALPHA
            const T alpha = nAlphas == 1 ? 0.5 : static_cast<T>(a) / static_cast<T>(nAlphas > 1 ? nAlphas - 1 : 1);

            // Setup Lambda
            if(lambdatype==LAMBDATYPEONE) lambdas[0]=lambdaarrayofa[a];

            /////////////
            //
            // SETUP FOLD (and weights)
            //
            ////////////
            // FOLDTYPE: 0 = any portion and can be overlapping
            // FOLDTYPE: 1 = non-overlapping folds
#define FOLDTYPE 1
            T fractrain;
            if(FOLDTYPE==0){
              fractrain=(nFolds>1 ? 0.8: 1.0);
            }
            else{
              fractrain=(nFolds>1 ? 1.0-1.0/((double)nFolds) : 1.0);
            }
            T fracvalid=1.0 - fractrain;
            T weights[mTrain];
            if(nFolds>1){
              for(unsigned int j=0;j<mTrain;++j){
                T foldon=1;
                int jfold=j-fi*fracvalid*mTrain;
                if(jfold>=0 && jfold<fracvalid*mTrain) foldon=1E-13;
                
                weights[j] = foldon*trainW[j];
                //              fprintf(stderr,"a=%d fold=%d j=%d foldon=%g trainW=%g weights=%g\n",a,fi,j,foldon,trainW[j],weights[j]); fflush(stderr);
              }
            }
            else{// then assume meant one should just copy weights
              for(unsigned int j=0;j<mTrain;++j) weights[j] = trainW[j];
            }

            // normalize weights before input (method has issue with small typical weights, so avoid normalization and just normalize in error itself only)
            if(0){
              T sumweight=0;
              for(unsigned int j=0;j<mTrain;++j) sumweight+=weights[j];
              if(sumweight!=0.0){
                for(unsigned int j=0;j<mTrain;++j) weights[j]/=sumweight;
              }
              else continue; // skip this fi,a
              //            fprintf(stderr,"a=%d fold=%d sumweights=%g\n",a,fi,sumweight); fflush(stderr);
            }
          

            /////////////////////
            //
            // Setup Solve
            //
            //////////////////////
            // setup f,g as functions of alpha
            std::vector <FunctionObj<T>> f;
            std::vector <FunctionObj<T>> g;
            f.reserve(mTrain);
            g.reserve(n);
            // minimize ||Ax-b||_2^2 + \alpha\lambda||x||_1 + (1/2)(1-alpha)*lambda x^2
            for (unsigned int j = 0; j < mTrain; ++j) f.emplace_back(kSquare, 1.0, trainY[j], weights[j]); // h2oaiglm.R
            //for (unsigned int j = 0; j < mTrain; ++j) f.emplace_back(kSquare, 1.0, trainY[j], trainW[j]); // h2oaiglm.R
            for (unsigned int j = 0; j < n - intercept; ++j) g.emplace_back(kAbs);
            if (intercept) g.emplace_back(kZero);



            ////////////////////////////
            //
            // LOOP OVER LAMBDA
            //
            ///////////////////////////////
            vector<double> scoring_history;
            int gotX0=0;
            double jump=DBL_MAX;
            double norm=(mValid==0 ? sdTrainY : sdValidY);
            int skiplambdaamount=0;
            int i;
            double trainRMSE=-1;
            double ivalidRMSE = -1;
            double validRMSE = -1;
            double tol0=1E-2; // highest acceptable tolerance (USER parameter)  Too high and won't go below standard deviation.
            double tol=tol0;
            T lambda=-1;
            double tbestalpha,tbestlambda,tbesttol=std::numeric_limits<double>::max(),tbestrmse[NUMRMSE];
            RMSELOOP(ri) tbestrmse[ri]=std::numeric_limits<double>::max();

            // LOOP over lambda
            for (i = 0; i < nlambdalocal; ++i) {
              if (flag) {
                continue;
              }

              // Set Lambda
              lambda = lambdas[i];
              DEBUG_FPRINTF(fil, "lambda %d = %f\n", i, lambda);


              //////////////
              //
              // if lambda path, control how go along path
              //
              //////////////
              if(lambdatype==LAMBDATYPEPATH){
              
                // Reset Solution if starting fresh for this alpha
                if(i==0){
                  // see if have previous solution for new alpha for better warmstart
                  if(gotpreviousX0){
                    //              DEBUG_FPRINTF(stderr,"m=%d a=%d i=%d Using old alpha solution\n",me,a,i);
                    //              for(unsigned int ll=0;ll<n;ll++) DEBUG_FPRINTF(stderr,"X0[%d]=%g\n",ll,X0[ll]);
                    h2oaiglm_data.SetInitX(X0);
                    h2oaiglm_data.SetInitLambda(L0);
                  }
                  else{
                    h2oaiglm_data.ResetX(); // reset X if new alpha if expect much different solution
                  }
                }

                ///////////////////////
                //
                // Set tolerances more automatically
                // (NOTE: that this competes with stopEarly() below in a good way so that it doesn't stop overly early just because errors are flat due to poor tolerance).
                // Note currently using jump or jumpuse.  Only using scoring vs. standard deviation.
                // To check total iteration count, e.g., : grep -a "Iter  :" output.txt|sort -nk 3|awk '{print $3}' | paste -sd+ | bc
                double jumpuse=DBL_MAX;
                tol=tol0;
                h2oaiglm_data.SetRelTol(tol); // set how many cuda devices to use internally in h2oaiglm
                h2oaiglm_data.SetAbsTol(0.5*tol); // set how many cuda devices to use internally in h2oaiglm
                h2oaiglm_data.SetMaxIter(1000);
                // see if getting below stddev, if so decrease tolerance
                if(scoring_history.size()>=1){
                  double ratio = (norm-scoring_history.back())/norm;

                  if(ratio>0.0){
                    double factor=0.05; // rate factor (USER parameter)
                    double tollow=1E-3; //lowest allowed tolerance (USER parameter)
                    tol = tol0*pow(2.0,-ratio/factor);
                    if(tol<tollow) tol=tollow;
                
                    h2oaiglm_data.SetRelTol(tol);
                    h2oaiglm_data.SetAbsTol(0.5*tol);
                    h2oaiglm_data.SetMaxIter(1000);
                    jumpuse=jump;
                  }
                  //              fprintf(stderr,"me=%d a=%d i=%d jump=%g jumpuse=%g ratio=%g tol=%g norm=%g score=%g\n",me,a,i,jump,jumpuse,ratio,tol,norm,scoring_history.back());
                }
              }
              else{// single lambda
                // assume warm-start value of X and other internal variables
                //                fprintf(stderr,"tol to use for last alpha=%g lambda=%g is %g\n",alphaarrayofa[a],lambdaarrayofa[a],tolarrayofa[a]); fflush(stderr);
                tol = tolarrayofa[a];
                //                tol = tol0;
                h2oaiglm_data.SetRelTol(tol);
                h2oaiglm_data.SetAbsTol(0.5*tol);
                h2oaiglm_data.SetMaxIter(1000);
              }



              ////////////////////
              //
              // Solve
              //
              ////////////////////
            
              DEBUG_FPRINTF(fil, "Starting to solve at %21.15g\n", timer<double>());
              T penalty_factor = static_cast<T>(1.0); // like h2oaiglm.R
              // assign lambda (no penalty for intercept, the last coeff, if present)
              for (unsigned int j = 0; j < n - intercept; ++j) {
                g[j].c = static_cast<T>(alpha * lambda * penalty_factor); //for L1
                g[j].e = static_cast<T>((1.0 - alpha) * lambda * penalty_factor); //for L2
              }
              if (intercept) {
                g[n - 1].c = 0;
                g[n - 1].e = 0;
              }
              // Solve
              h2oaiglm_data.Solve(f, g);


              
              int doskiplambda=0;
              if(lambdatype==LAMBDATYPEPATH){
                /////////////////
                //
                // Check if getting solution was too easy and was 0 iterations.  If so, overhead is not worth it, so try skipping by 1.
                //
                /////////////////
                if(h2oaiglm_data.GetFinalIter()==0){
                  doskiplambda=1;
                  skiplambdaamount++;
                }
                else{
                  // reset if not 0 iterations
                  skiplambdaamount=0;
                }

            
                ////////////////////////////////////////////
                //
                // Check if solution was found
                //
                ////////////////////////////////////////////
                int maxedout=0;
                if(h2oaiglm_data.GetFinalIter()==h2oaiglm_data.GetMaxIter()) maxedout=1;
                else maxedout=0;

                if(maxedout) h2oaiglm_data.ResetX(); // reset X if bad solution so don't start next lambda with bad solution
                // store good high-lambda solution to start next alpha with (better than starting with low-lambda solution)
                if(gotX0==0 && maxedout==0){
                  gotX0=1;
                  // TODO: FIXME: Need to get (and have solver set) best solution or return all, because last is not best.
                  gotpreviousX0=1;
                  memcpy(X0,&h2oaiglm_data.GetX()[0],n*sizeof(T));
                  memcpy(L0,&h2oaiglm_data.GetLambda()[0],mTrain*sizeof(T));
                }

              }

              if (intercept) {
                DEBUG_FPRINTF(fil, "intercept: %g\n", h2oaiglm_data.GetX()[n - 1]);
                DEBUG_FPRINTF(stdout, "intercept: %g\n", h2oaiglm_data.GetX()[n - 1]);
              }


              ////////////////////////////////////////
              //
              // Get predictions for training and validation
              //
              //////////////////////////////////////////

              // Degrees of freedom
              size_t dof = 0;
              {
                for (size_t i = 0; i < n - intercept; ++i) {
                  if (std::abs(h2oaiglm_data.GetX()[i]) > 1e-8) {
                    dof++;
                  }
                }
              }



              // TRAIN PREDS
#if(OLDPRED)
              std::vector <T> trainPreds(mTrain);
              for (size_t i = 0; i < mTrain; ++i) {
                trainPreds[i] = 0;
                for (size_t j = 0; j < n; ++j) {
                  trainPreds[i] += h2oaiglm_data.GetX()[j] * trainX[i * n + j]; //add predictions
                }
              }
#else
              std::vector <T> trainPreds(&h2oaiglm_data.GettrainPreds()[0], &h2oaiglm_data.GettrainPreds()[0]+mTrain);
#endif
              // RMSE: TRAIN
              trainRMSE = h2oaiglm::getRMSE(weights, mTrain, &trainPreds[0], trainY);
              if(standardize){
                trainRMSE *= sdTrainY;
                for (size_t i = 0; i < mTrain; ++i) {
                  // reverse standardization
                  trainPreds[i]*=sdTrainY; //scale
                  trainPreds[i]+=meanTrainY; //intercept
                  //assert(trainPreds[i] == h2oaiglm_data.GetY()[i]); //FIXME: CHECK
                }
              }

            
              // RMSE: on fold's held-out training data
              if(nFolds>1){
                const T offset=1.0;
                ivalidRMSE = h2oaiglm::getRMSE(offset, weights, mTrain, &trainPreds[0], trainY);
              }
              else{
                ivalidRMSE = -1.0;
              }


              // VALID (preds and rmse)
              validRMSE = -1;
              if (mValid > 0) {

                T weightsvalid[mValid];
                for (size_t i = 0; i < mValid; ++i) {//row
                  weightsvalid[i] = 1.0;
                }

                // Valid Preds
#if(OLDPRED)
                std::vector <T> validPreds(mValid);
                for (size_t i = 0; i < mValid; ++i) { //row
                  validPreds[i] = 0;
                  for (size_t j = 0; j < n; ++j) { //col
                    validPreds[i] += h2oaiglm_data.GetX()[j] * validX[i * n + j]; //add predictions
                  }
                }
#else
                std::vector <T> validPreds(&h2oaiglm_data.GetvalidPreds()[0], &h2oaiglm_data.GetvalidPreds()[0]+mValid);
#endif
                // RMSE: VALIDs
                validRMSE = h2oaiglm::getRMSE(weightsvalid,mValid, &validPreds[0], validY);
                if(standardize){
                  validRMSE *= sdTrainY;
                  for (size_t i = 0; i < mValid; ++i) { //row
                    // reverse (fitted) standardization
                    validPreds[i]*=sdTrainY; //scale
                    validPreds[i]+=meanTrainY; //intercept
                  }
                }
              }

              ////////////
              //
              // report scores
              //
              ////////////
              Printmescore(fil);
              Printmescoresimple(fillatest);
              Printmescore(stdout);

              T localrmse[NUMRMSE];
              localrmse[0]=trainRMSE;
              localrmse[1]=ivalidRMSE;
              localrmse[2]=validRMSE;
              if(tbestrmse[iwhichrmse]>localrmse[iwhichrmse]){
                tbestalpha=alpha;
                tbestlambda=lambda;
                tbesttol=tol;
                RMSELOOP(ri) tbestrmse[ri]=localrmse[ri];
              }

              // save scores
              scoring_history.push_back(localrmse[iwhichrmse]);

              if(lambdatype==LAMBDATYPEPATH){
                if(fi==0 && givefullpath){ // only store first fold for user
                  //#define MAPXALL(i,a,which) (which + a*(n+NUMRMSE+NUMOTHER) + i*(n+NUMRMSE+NUMOTHER)*nLambdas)
                  //#define MAPXBEST(a,which) (which + a*(n+NUMRMSE+NUMOTHER))
                  //#define NUMOTHER 3 // for lambda, alpha, tol
                  // Save solution to return to user
                  memcpy( &((*Xvsalphalambda)[MAPXALL(i,a,0)]),&h2oaiglm_data.GetX()[0],n);
                  // Save rmse to return to user
                  RMSELOOP(ri) (*Xvsalphalambda)[MAPXALL(i,a,n+ri)] = localrmse[ri];
                  // Save lambda to return to user
                  (*Xvsalphalambda)[MAPXALL(i,a,n+NUMRMSE)] = lambda;
                  // Save alpha to return to user
                  (*Xvsalphalambda)[MAPXALL(i,a,n+NUMRMSE+1)] = alpha;
                  // Save tol to return to user
                  (*Xvsalphalambda)[MAPXALL(i,a,n+NUMRMSE+2)] = tol;
                }
              }
              else{ // only save here if doing nFolds>2
                memcpy( &((*Xvsalpha)[MAPXBEST(a,0)]),&h2oaiglm_data.GetX()[0],n);
                // Save rmse to return to user
                RMSELOOP(ri) (*Xvsalpha)[MAPXBEST(a,n+ri)] = localrmse[ri];
                // Save lambda to return to user
                (*Xvsalpha)[MAPXBEST(a,n+NUMRMSE)] = lambda;
                // Save alpha to return to user
                (*Xvsalpha)[MAPXBEST(a,n+NUMRMSE+1)] = alpha;
                // Save tol to return to user
                (*Xvsalpha)[MAPXBEST(a,n+NUMRMSE+2)] = tol;
              }



              if(lambdatype==LAMBDATYPEPATH){
                if(DOSTOPEARLY){
                  if(scoring_history.size()>=1){
                    double ratio = (norm-scoring_history.back())/norm;

                    double fracdof=0.5; //USER parameter.
                    if(ratio>0.0 && (double)dof>fracdof*(double)(n)){ // only consider stopping if explored most degrees of freedom, because at dof~0-1 error can increase due to tolerance in solver.
                      //                  fprintf(stderr,"ratio=%g dof=%zu fracdof*n=%g\n",ratio,dof,fracdof*n); fflush(stderr);
                      // STOP EARLY CHECK
                      int k = 3; //TODO: ask the user for this parameter
                      double tolerance = 0.0; // stop when not improved over 3 successive lambdas (averaged over window 3) // NOTE: Don't use tolerance=0 because even for simple.txt test this stops way too early when error is quite high
                      bool moreIsBetter = false;
                      bool verbose = true;
                      if (stopEarly(scoring_history, k, tolerance, moreIsBetter, verbose,norm,&jump)) {
                        break;
                      }
                    }
                  }
                }

                // if can skip over lambda, do so, but still print out the score as if constant for new lambda
                if(doskiplambda){
                  for (int ii = 0; ii < skiplambdaamount; ++ii) {
                    i++;
                    lambda = lambdas[i];
                    Printmescore(fil);
                    Printmescoresimple(fillatest);
                    Printmescore(stdout);
                  }
                }
              }
            
            }// over lambda(s)


            // store results
            int pickfi;
            if(lambdatype==LAMBDATYPEPATH) pickfi=fi; // variable lambda folds
            else pickfi=nFolds+fi; // fixed-lambda folds
            // store RMSE (thread-safe)
            alphaarray[pickfi][a]=tbestalpha;
            lambdaarray[pickfi][a]=tbestlambda;
            tolarray[pickfi][a]=tbesttol;
            RMSELOOP(ri) rmsearray[ri][pickfi][a]=tbestrmse[ri];

            // if not doing folds, store best solution over all lambdas
            if(lambdatype==LAMBDATYPEPATH && nFolds<2){
              if(fi==0){ // only store first fold for user
                memcpy( &((*Xvsalpha)[MAPXBEST(a,0)]),&h2oaiglm_data.GetX()[0],n); // not quite best, last lambda TODO FIXME
                // Save rmse to return to user
                RMSELOOP(ri) (*Xvsalpha)[MAPXBEST(a,n+ri)] = tbestrmse[ri];
                // Save lambda to return to user
                (*Xvsalpha)[MAPXBEST(a,n+NUMRMSE)] = tbestlambda;
                // Save alpha to return to user
                (*Xvsalpha)[MAPXBEST(a,n+NUMRMSE+1)] = tbestalpha;
                // Save tol to return to user
                (*Xvsalpha)[MAPXBEST(a,n+NUMRMSE+2)] = tbesttol;
              }
            }


          }// over folds
        }// over alpha


#pragma omp barrier // barrier so alphaarray, lambdaarray, rmsearray are filled and ready to be read by all threads
        int fistart;
        if(lambdatype==LAMBDATYPEPATH) fistart=0; // variable lambda folds
        else fistart=nFolds; // fixed-lambda folds


        // get CV averaged RMSE and best solution (using shared memory arrays that are thread-safe)
        double bestalpha=0;
        double bestlambda=0;
        double besttol=std::numeric_limits<double>::max();
        double bestrmse[NUMRMSE];
        RMSELOOP(ri) bestrmse[ri]=std::numeric_limits<double>::max();
        for (size_t a = 0; a < nAlphas; ++a) { //alpha
          alphaarrayofa[a]=0.0;
          lambdaarrayofa[a]=0.0;
          tolarrayofa[a]=std::numeric_limits<double>::max();
          RMSELOOP(ri) rmsearrayofa[ri][a]=0.0;
          for (size_t fi = fistart; fi < fistart+nFolds; ++fi) { //fold
            alphaarrayofa[a]+=alphaarray[fi][a];
            lambdaarrayofa[a]+=lambdaarray[fi][a];
#define MIN(a,b) ((a)<(b)?(a):(b))
            tolarrayofa[a]=MIN(tolarrayofa[a],tolarray[fi][a]); // choose common min tolerance
            RMSELOOP(ri) rmsearrayofa[ri][a]+=rmsearray[ri][fi][a];
          }
          // get average rmse over folds for this alpha
          alphaarrayofa[a]/=((double)(nFolds));
          lambdaarrayofa[a]/=((double)(nFolds));
          RMSELOOP(ri) rmsearrayofa[ri][a]/=((double)(nFolds));
          if(rmsearrayofa[owhichrmse][a]<bestrmse[owhichrmse]){
            bestalpha=alphaarrayofa[a]; // get alpha for this case
            bestlambda=lambdaarrayofa[a]; // get lambda for this case
            besttol=tolarrayofa[a]; // get tol for this case
            RMSELOOP(ri) bestrmse[ri]=rmsearrayofa[ri][a]; // get best rmse as average for this alpha
          }
          if(lambdatype==LAMBDATYPEPATH) fprintf(stderr,"To use for last CV models: alpha=%g lambda=%g tol=%g\n",alphaarrayofa[a],lambdaarrayofa[a],tolarrayofa[a]); fflush(stderr);
        }


        
        // print result (all threads have same result, so only need to print on one thread)
        if(me==0) PrintmescoresimpleCV(stdout,lambdatype,bestalpha,bestlambda,bestrmse[0],bestrmse[1],bestrmse[2]);
          
      }// over lambdatype

      if(X0) delete [] X0;
      if(L0) delete [] L0;
      if (fil != NULL) fclose(fil);
    } // end parallel region


    ///////////////////////
    //
    // report over all folds, cross-validated model, and over alphas
    //
    ///////////////////////
    for (size_t fi = 0; fi < totalfolds; ++fi) { //fold
      for (size_t a = 0; a < nAlphas; ++a) { //alpha
        fprintf(stderr,"pass=%d fold=%zu alpha=%21.15g lambda=%21.15g rmseTrain=%21.15g rmseiValid=%21.15g rmseValid=%21.15g\n",(fi>=nFolds ? 1 : 0),(fi>=nFolds ? fi-nFolds : fi),alphaarray[fi][a],lambdaarray[fi][a],rmsearray[0][fi][a],rmsearray[1][fi][a],rmsearray[2][fi][a]); fflush(stderr);
      }
    }


    // free any malloc's
    if(trainX && OLDPRED) free(trainX);
    if(trainY) free(trainY);
    if(validX && OLDPRED) free(validX);
    if(validY) free(validY);
    if(trainW) free(trainW);

    double tf = timer<double>();
    fprintf(stdout, "END SOLVE: type 1 mTrain %d n %d mValid %d twall %g tsolve(post-dataongpu) %g\n", (int) mTrain, (int) n,   (int) mValid, tf - t, tf - t1me0);
    if (flag) {
      fprintf(stderr, "Signal caught. Terminated early.\n"); fflush(stderr);
      flag = 0; // set flag
    }
    return tf - t;
  }


  template double ElasticNetptr<double>(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                                        size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                                        double lambda_min_ratio, int nLambdas, int nFolds, int nAlphas,
                                        void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr);

  template double ElasticNetptr<float>(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                                       size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                                       double lambda_min_ratio, int nLambdas, int nFolds, int nAlphas,
                                       void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr);



  
#ifdef __cplusplus
  extern "C" {
#endif

    double elastic_net_ptr_double(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                                  size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                                  double lambda_min_ratio, int nLambdas, int nFolds, int nAlphas,
                                  void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr) {
      return ElasticNetptr<double>(sourceDev, datatype, sharedA, nThreads, nGPUs, ord,
                                   mTrain, n, mValid, intercept, standardize,
                                   lambda_min_ratio, nLambdas, nFolds, nAlphas,
                                   trainXptr, trainYptr, validXptr, validYptr, weightptr);
    }
    double elastic_net_ptr_float(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                                 size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                                 double lambda_min_ratio, int nLambdas, int nFolds, int nAlphas,
                                 void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr) {
      return ElasticNetptr<float>(sourceDev, datatype, sharedA, nThreads, nGPUs, ord,
                                  mTrain, n, mValid, intercept, standardize,
                                  lambda_min_ratio, nLambdas, nFolds, nAlphas,
                                  trainXptr, trainYptr, validXptr, validYptr, weightptr);
    }

#ifdef __cplusplus
  }
#endif
}
