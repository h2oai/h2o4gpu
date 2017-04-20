#include "elastic_net_mapd.h"

#define OLDPRED 0

using namespace std;

namespace pogs {

  bool stopEarly(vector<double> val, int k, double tolerance, bool moreIsBetter, bool verbose) {
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

    // check if any of the moving averages is better than the reference (by at least tolerance relative improvement)
    double ref = moving_avg[0];
    bool improved = false;
    for (int i=1;i<moving_avg.size();++i) {
      if (moreIsBetter)
        improved |= (moving_avg[i] > ref*(1+tolerance));
      else
        improved |= (moving_avg[i] < ref*(1-tolerance));
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
// See <pogs>/matlab/examples/lasso_path.m for detailed description.
// m and n are training data size
    template<typename T>
    double ElasticNetptr(int sourceDev, int datatype, int nGPUs, const char ord,
                         size_t mTrain, size_t n, size_t mValid, int intercept, int standardize, double lambda_max0,
                         double lambda_min_ratio, int nLambdas, int nAlphas,
                         double sdTrainY, double meanTrainY,
                         void *trainXptr, void *trainYptr, void *validXptr, void *validYptr) {

      int nlambda = nLambdas;
      if (nlambda <= 1) {
        cerr << "Must use nlambda > 1\n";
        exit(-1);
      }


      // number of openmp threads = number of cuda devices to use
#ifdef _OPENMP
      int omt=omp_get_max_threads();
#define MIN(a,b) ((a)<(b)?(a):(b))
      omp_set_num_threads(MIN(omt,nGPUs));
      int nth=omp_get_max_threads();
      nGPUs=nth; // openmp threads = cuda devices used
      cout << "Number of original threads=" << omt << " Number of threads for cuda=" << nth << endl;

#else
#error Need OpenMP
#endif


      // for source, create class objects that creates cuda memory, cpu memory, etc.
      // This takes-in raw GPU pointer
      //  pogs::MatrixDense<T> Asource_(sourceDev, ord, mTrain, n, mValid, reinterpret_cast<T *>(trainXptr));
      pogs::MatrixDense<T> Asource_(sourceDev, datatype, ord, mTrain, n, mValid,
                                    reinterpret_cast<T *>(trainXptr), reinterpret_cast<T *>(trainYptr),
                                    reinterpret_cast<T *>(validXptr), reinterpret_cast<T *>(validYptr));
      // now can always access A_(sourceDev) to get pointer from within other MatrixDense calls


      // temporarily get trainX, etc. from pogs (which may be on gpu)
      T *trainX=NULL;
      T *trainY=NULL;
      T *validX=NULL;
      T *validY=NULL;
      if(OLDPRED) trainX = (T *) malloc(sizeof(T) * mTrain * n);
      trainY = (T *) malloc(sizeof(T) * mTrain);
      if(OLDPRED) validX = (T *) malloc(sizeof(T) * mValid * n);
      validY = (T *) malloc(sizeof(T) * mValid);

      if(OLDPRED) Asource_.GetTrainX(datatype, mTrain * n, &trainX);
      Asource_.GetTrainY(datatype, mTrain, &trainY);
      if(OLDPRED) Asource_.GetValidX(datatype, mValid * n, &validX);
      Asource_.GetValidY(datatype, mValid, &validY);


      // Setup each thread's pogs
      double t = timer<double>();
#pragma omp parallel
      {
        int me = omp_get_thread_num();

        char filename[100];
        sprintf(filename, "me%d.%s.txt", me, _GITHASH_);
        FILE *fil = fopen(filename, "wt");
        if (fil == NULL) {
          cerr << "Cannot open filename=" << filename << endl;
          exit(0);
        }

        double t0 = timer<double>();
        fprintf(fil, "Moving data to the GPU. Starting at %21.15g\n", t0);
        fflush(fil);
        // create class objects that creates cuda memory, cpu memory, etc.
        pogs::MatrixDense<T> A_(me, Asource_);
        pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(me, A_);
        double t1 = timer<double>();
        fprintf(fil, "Done moving data to the GPU. Stopping at %21.15g\n", t1);
        fprintf(fil, "Done moving data to the GPU. Took %g secs\n", t1 - t0);
        fflush(fil);

        pogs_data.SetnDev(1); // set how many cuda devices to use internally in pogs
//    pogs_data.SetRelTol(1e-4); // set how many cuda devices to use internally in pogs
//    pogs_data.SetAbsTol(1e-5); // set how many cuda devices to use internally in pogs
//    pogs_data.SetAdaptiveRho(true);
        //pogs_data.SetEquil(false);
//    pogs_data.SetRho(1);
//    pogs_data.SetVerbose(5);
//    pogs_data.SetMaxIter(20000);

        int N = nAlphas; // number of alpha's
        if (N % nGPUs != 0) {
          fprintf(stderr,
                  "NOTE: Number of alpha's not evenly divisible by number of GPUs, so not efficint use of GPUs.\n");
          fflush(stderr);
        }
        fprintf(fil, "BEGIN SOLVE\n");
        fflush(fil);
        int a;


        fprintf(stderr, "lambda_max0: %f\n", lambda_max0);
        fflush(stderr);
#pragma omp for
        for (a = 0; a < N; ++a) { //alpha search
          const T alpha = N == 1 ? 0.5 : static_cast<T>(a) / static_cast<T>(N > 1 ? N - 1 : 1);
          const T lambda_min = lambda_min_ratio * static_cast<T>(lambda_max0); // like pogs.R
          T lambda_max = lambda_max0 / std::max(static_cast<T>(1e-2), alpha); // same as H2O
          if (alpha == 1 && mTrain > 10000) {
            lambda_max *= 2;
            lambda_min_ratio /= 2;
          }
          fprintf(stderr, "lambda_max: %f\n", lambda_max);
          fprintf(stderr, "lambda_min: %f\n", lambda_min);
          fflush(stderr);
          fprintf(fil, "lambda_max: %f\n", lambda_max);
          fprintf(fil, "lambda_min: %f\n", lambda_min);
          fflush(fil);

          // setup f,g as functions of alpha
          std::vector <FunctionObj<T>> f;
          std::vector <FunctionObj<T>> g;
          f.reserve(mTrain);
          g.reserve(n);
          // minimize ||Ax-b||_2^2 + \alpha\lambda||x||_1 + (1/2)(1-alpha)*lambda x^2
          T penalty_factor = static_cast<T>(1.0); // like pogs.R
          //T weights = static_cast<T>(1.0);// / (static_cast<T>(mTrain)));

          for (unsigned int j = 0; j < mTrain; ++j) f.emplace_back(kSquare, 1.0, trainY[j]); // pogs.R
          for (unsigned int j = 0; j < n - intercept; ++j) g.emplace_back(kAbs);
          if (intercept) g.emplace_back(kZero);

          // Regularization path: geometric series from lambda_max to lambda_min
          std::vector <T> lambdas(nlambda);
          double dec = std::pow(lambda_min_ratio, 1.0 / (nlambda - 1.));
          lambdas[0] = lambda_max;
          for (int i = 1; i < nlambda; ++i)
            lambdas[i] = lambdas[i - 1] * dec;

          fprintf(fil, "alpha%f\n", alpha);
          vector<double> scoring_history;
          for (int i = 0; i < nlambda; ++i) {
            T lambda = lambdas[i];
            fprintf(fil, "lambda %d = %f\n", i, lambda);

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
            fprintf(fil, "Starting to solve at %21.15g\n", timer<double>());
            fflush(fil);
            if(i==0) pogs_data.ResetX(); // reset X if new alpha if expect much different solution
            pogs_data.Solve(f, g);
            if(pogs_data.GetFinalIter()==pogs_data.GetMaxIter()) pogs_data.ResetX(); // reset X if bad

            if (intercept) {
              fprintf(fil, "intercept: %g\n", pogs_data.GetX()[n - 1]);
              fprintf(stdout, "intercept: %g\n", pogs_data.GetX()[n - 1]);
              fflush(stdout);
            }

            size_t dof = 0;
            for (size_t i = 0; i < n - intercept; ++i) {
              if (std::abs(pogs_data.GetX()[i]) > 1e-8) {
                dof++;
              }
            }

#if(OLDPRED)
            std::vector <T> trainPreds(mTrain);
            for (size_t i = 0; i < mTrain; ++i) {
              trainPreds[i] = 0;
              for (size_t j = 0; j < n; ++j) {
                trainPreds[i] += pogs_data.GetX()[j] * trainX[i * n + j]; //add predictions
              }
            }
#else
            std::vector <T> trainPreds(&pogs_data.GettrainPreds()[0], &pogs_data.GettrainPreds()[0]+mTrain);
#endif
            double trainRMSE = getRMSE(mTrain, &trainPreds[0], trainY);
            if(standardize){
              trainRMSE *= sdTrainY;
              for (size_t i = 0; i < mTrain; ++i) {
                // reverse standardization
                trainPreds[i]*=sdTrainY; //scale
                trainPreds[i]+=meanTrainY; //intercept
                //assert(trainPreds[i] == pogs_data.GetY()[i]); //FIXME: CHECK
              }
            }
//        // DEBUG START
//        for (size_t j=0; j<n; ++j) {
//          cout << pogs_data.GetX()[j] << endl;
//        }
//        for (int i=0;i<mTrain;++i) {
//          for (int j=0;j<n;++j) {
//            cout << trainX[i*n+j] << " ";
//          }
//          cout << " -> " << trainY[i] << "\n";
//          cout << "\n";
//        }
//        // DEBUG END

            double validRMSE = -1;
            if (mValid > 0) {
#if(OLDPRED)
              std::vector <T> validPreds(mValid);
              for (size_t i = 0; i < mValid; ++i) { //row
                validPreds[i] = 0;
                for (size_t j = 0; j < n; ++j) { //col
                  validPreds[i] += pogs_data.GetX()[j] * validX[i * n + j]; //add predictions
                }
              }
#else
              std::vector <T> validPreds(&pogs_data.GetvalidPreds()[0], &pogs_data.GetvalidPreds()[0]+mValid);
#endif
              validRMSE = getRMSE(mValid, &validPreds[0], validY);
              if(standardize){
                validRMSE *= sdTrainY;
                for (size_t i = 0; i < mValid; ++i) { //row
                  // reverse (fitted) standardization
                  validPreds[i]*=sdTrainY; //scale
                  validPreds[i]+=meanTrainY; //intercept
                }
              }
            }

            fprintf(fil, "%s.me: %d a: %d alpha: %g intercept: %d standardize: %d i: %d lambda: %g dof: %d trainRMSE: %f validRMSE: %f\n", _GITHASH_, me, a, alpha,intercept,standardize, (int)i, lambda, (int)dof, trainRMSE, validRMSE);
            fflush(fil);
            fprintf(stdout, "%s.me: %d a: %d alpha: %g intercept: %d standardize: %d i: %d lambda: %g dof: %d trainRMSE: %f validRMSE: %f\n", _GITHASH_, me, a, alpha,intercept,standardize, (int)i, lambda, (int)dof, trainRMSE, validRMSE);
            fflush(stdout);
            int k = 3; //TODO: ask the user for this parameter
            scoring_history.push_back(mValid > 0 ? validRMSE : trainRMSE);
            double tolerance = 0; // stop when not improved over 3 successive lambdas (averaged over window 3)
            bool moreIsBetter = false;
            bool verbose = true;
            if (stopEarly(scoring_history, k, tolerance, moreIsBetter, verbose)) {
              break;
            }
          }// over lambda
        }// over alpha
        if (fil != NULL) fclose(fil);
      } // end parallel region


      // free any malloc's
      if(trainX && OLDPRED) free(trainX);
      if(trainY) free(trainY);
      if(validX && OLDPRED) free(validX);
      if(validY) free(validY);

      double tf = timer<double>();
      fprintf(stdout, "END SOLVE: type 1 mTrain %d n %d mValid %d twall %g\n", (int) mTrain, (int) n,
              (int) mValid, tf - t);
      return tf - t;
    }

    template double ElasticNetptr<double>(int sourceDev, int datatype, int nGPUs, const char ord,
                                          size_t mTrain, size_t n, size_t mValid, int intercept, int standardize, double lambda_max0,
                                          double lambda_min_ratio, int nLambdas, int nAlphas,
                                          double sdTrainY, double meanTrainY,
                                          void *trainXptr, void *trainYptr, void *validXptr, void *validYptr);

    template double ElasticNetptr<float>(int sourceDev, int datatype, int nGPUs, const char ord,
                                         size_t mTrain, size_t n, size_t mValid, int intercept, int standardize, double lambda_max0,
                                         double lambda_min_ratio, int nLambdas, int nAlphas,
                                         double sdTrainY, double meanTrainY,
                                         void *trainXptr, void *trainYptr, void *validXptr, void *validYptr);

#ifdef __cplusplus
    extern "C" {
#endif

    int make_ptr_double(int sourceDev, size_t mTrain, size_t n, size_t mValid,
                        double* trainX, double* trainY, double* validX, double* validY,
                        void**a, void**b, void**c, void**d) {
      return makePtr<double>(sourceDev, mTrain, n, mValid, trainX, trainY, validX, validY, a, b, c, d);
    }
    int make_ptr_float(int sourceDev, size_t mTrain, size_t n, size_t mValid,
                       float* trainX, float* trainY, float* validX, float* validY,
                       void**a, void**b, void**c, void**d) {
      return makePtr<float>(sourceDev, mTrain, n, mValid, trainX, trainY, validX, validY, a, b, c, d);
    }
    double elastic_net_ptr_double(int sourceDev, int datatype, int nGPUs, int ord,
                                  size_t mTrain, size_t n, size_t mValid, int intercept, int standardize, double lambda_max0,
                                  double lambda_min_ratio, int nLambdas, int nAlphas,
                                  double sdTrainY, double meanTrainY,
                                  void *trainXptr, void *trainYptr, void *validXptr, void *validYptr) {
      return ElasticNetptr<double>(sourceDev, datatype, nGPUs, ord==1?'r':'c', mTrain, n, mValid,
                                   intercept, standardize, lambda_max0, lambda_min_ratio, nLambdas, nAlphas, sdTrainY, sourceDev,
                                   trainXptr, trainYptr, validXptr, validYptr);
    }
    double elastic_net_ptr_float(int sourceDev, int datatype, int nGPUs, int ord,
                                 size_t mTrain, size_t n, size_t mValid, int intercept, int standardize, double lambda_max0,
                                 double lambda_min_ratio, int nLambdas, int nAlphas,
                                 double sdTrainY, double meanTrainY,
                                 void *trainXptr, void *trainYptr, void *validXptr, void *validYptr) {
      return ElasticNetptr<float>(sourceDev, datatype, nGPUs, ord==1?'r':'c', mTrain, n, mValid,
                                  intercept, standardize, lambda_max0, lambda_min_ratio, nLambdas, nAlphas, sdTrainY, sourceDev,
                                  trainXptr, trainYptr, validXptr, validYptr);
    }

#ifdef __cplusplus
    }
#endif
}
