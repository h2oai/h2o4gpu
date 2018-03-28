/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "elastic_net_ptr.h"
#include <float.h>
#include "../include/util.h"
#include <sys/stat.h>

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
		printf("Failed to run command\n");
		return "NA";
	}
	fgets(c, sizeof(c) - 1, fp); //only take the first line
	res += c;
	pclose(fp);
	res.erase(std::remove(res.begin(), res.end(), '\n'), res.end());
	//  cout << "cmd: " << res << endl;
	return res;
}

const std::string CPUTYPE =
		cmd(
				"lscpu | grep 'Model name' | cut -d: -f2- | sed 's/ \\+//g' | sed 's/Intel(R)//' | sed 's/Core(TM)//' | sed 's/CPU//'");
const std::string SOCKETS = cmd(
		"lscpu | grep 'Socket(s)' | cut -d: -f2- | sed 's/ \\+//g'");

const std::string GPUTYPE =
		cmd(
				"nvidia-smi -q | grep 'Product Name' | cut -d: -f2- | sed 's/ \\+//g' | tail -n 1");
const std::string NGPUS =
		cmd(
				"nvidia-smi -q | grep 'Product Name' | cut -d: -f2- | sed 's/ \\+//g' | wc -l");

#ifdef HAVECUDA
const std::string HARDWARE = NGPUS + "x" + GPUTYPE;
#else
const std::string HARDWARE = SOCKETS + "x" + CPUTYPE;
#endif

#define VERBOSEENET 0
#define VERBOSEANIM 0

#if(VERBOSEENET)
#define Printmescore(thefile)  fprintf(thefile,                         \
                                       "%s.me: %d ARCH: %s:%s BLAS: %s%d COMP: %s sharedA: %d nThreads: %d nGPUs: %d time: %21.15g lambdatype: %d fi: %d a: %d alpha: %g intercept: %d standardize: %d i: %d " \
                                       "lambda: %g dof: %zu trainError: %f ivalidError: %f validError: %f ", \
                                       _GITHASH_, me, TEXTARCH, HARDWARE.c_str(), TEXTBLAS, blasnumber, TEXTCOMP, sharedA, nThreads, nGPUs, timer<double>(), lambdatype, fi, a, alpha,intercept,standardize, (int)i, \
                                       lambda, dof, trainError, ivalidError, validError);for(int lll=0;lll<NUMBETA;lll++) fprintf(thefile,"%d %21.15g ",whichbeta[lll],valuebeta[lll]); fprintf(thefile,"\n"); fflush(thefile);
#define Printmescore_predict(thefile)  fprintf(thefile,                         \
                                       "%s.me: %d ARCH: %s:%s BLAS: %s%d COMP: %s sharedA: %d nThreads: %d nGPUs: %d time: %21.15g a: %d intercept: %d standardize: %d i: %d " \
                                       "validError: %f ", \
                                       _GITHASH_, me, TEXTARCH, HARDWARE.c_str(), TEXTBLAS, blasnumber, TEXTCOMP, sharedA, nThreads, nGPUs, timer<double>(), a,intercept,standardize, (int)i, \
                                       validError); fflush(thefile);
#define Printmescore_predictnovalid(thefile)  fprintf(thefile,                         \
                                       "%s.me: %d ARCH: %s:%s BLAS: %s%d COMP: %s sharedA: %d nThreads: %d nGPUs: %d time: %21.15g a: %d intercept: %d standardize: %d i: %d ", \
                                       _GITHASH_, me, TEXTARCH, HARDWARE.c_str(), TEXTBLAS, blasnumber, TEXTCOMP, sharedA, nThreads, nGPUs, timer<double>(), a,intercept,standardize, (int)i); fflush(thefile);
#else
#define Printmescore(thefile)
#define Printmescore_predict(thefile)
#define Printmescore_predictnovalid(thefile)
#endif

//#if(VERBOSEANIM)
#define Printmescoresimple(thefile)   fprintf(thefile,"%21.15g %d %d %d %d %21.15g %21.15g %21.15g %21.15g %21.15g\n", timer<double>(), lambdatype, fi, a, i, alpha, lambda, trainError, ivalidError, validError); fflush(thefile);
#define Printmescoresimple_predict(thefile)   fprintf(thefile,"%21.15g %d %d %21.15g\n", timer<double>(), a, i, validError); fflush(thefile);
#define Printmescoresimple_predictnovalid(thefile)   fprintf(thefile,"%21.15g %d %d\n", timer<double>(), a, i); fflush(thefile);
#define NUMBETA 10 // number of beta to report
#define Printmescoresimple2(thefile)  fprintf(thefile,"%21.15g %d %d %d %d %21.15g %21.15g %zu ", timer<double>(), lambdatype, fi, a, i, alpha, lambda, dof); for(int lll=0;lll<NUMBETA;lll++) fprintf(thefile,"%d %21.15g ",whichbeta[lll],valuebeta[lll]); fprintf(thefile,"\n"); fflush(thefile);
//#else
//#define Printmescoresimple(thefile)
//#define Printmescoresimple_predict(thefile)
//#define Printmescoresimple_predictnovalid(thefile)
//#define NUMBETA 10 // number of beta to report
//#define Printmescoresimple2(thefile)
//#endif

#if(VERBOSEENET)
#define PrintmescoresimpleCV(thefile,lambdatype,bestalpha,bestlambda,besterror1,besterror2,besterror3)  fprintf(thefile,"BEST: %21.15g %d %21.15g %21.15g %21.15g %21.15g %21.15g\n", timer<double>(), lambdatype, bestalpha, bestlambda, besterror1,besterror2,besterror3 ); fflush(thefile);
#else
#define PrintmescoresimpleCV(thefile,lambdatype,bestalpha,bestlambda,besterror1,besterror2,besterror3)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <signal.h> //  our new library

#define OLDPRED 0 // JONTODO: cleanup: if OLDPRED=1, then must set sharedAlocal=0 in examples/cpp/elastic_net_ptr_driver.cpp when doing make pointer part, so that don't overwrite original data (due to equilibration) so can be used for scoring.

#define RELAXEARLYSTOP 0

namespace h2o4gpu {

volatile sig_atomic_t flag = 0;
inline void my_function(int sig) { // can be called asynchronously
	fprintf(stderr, "Caught signal %d. Terminating shortly.\n", sig);
	flag = 1; // set flag
}

bool stopEarly(vector<double> val, int k, double tolerance, bool moreIsBetter,
		bool verbose, double norm, double *jump) {
	if (val.size() - 1 < 2 * k)
		return false; //need 2k scoring events (+1 to skip the very first one, which might be full of NaNs)
	vector<double> moving_avg(k + 1); //one moving avg for the last k+1 scoring events (1 is reference, k consecutive attempts to improve)

	// compute moving average(s)
	for (int i = 0; i < moving_avg.size(); ++i) {
		moving_avg[i] = 0;
		int startidx = val.size() - 2 * k + i;
		for (int j = 0; j < k; ++j)
			moving_avg[i] += val[startidx + j];
		moving_avg[i] /= k;
	}
	if (verbose) {
		cout << "JUnit: moving averages: ";
		copy(moving_avg.begin(), moving_avg.end(),
				ostream_iterator<double>(cout, " "));
		cout << endl;
	}

	// get average of moving average
	double moving_avgavg = 0;
	{
		int i;
		for (i = 0; i < moving_avg.size(); ++i) {
			moving_avgavg += moving_avg[i];
		}
		moving_avgavg /= ((double) i);
	}

	// var variance and error of moving average
	double var = 0, error = 0;
	{
		int i;
		for (i = 0; i < moving_avg.size(); ++i) {
			var += pow(moving_avg[i] - moving_avgavg, 2.0);
		}
		error = sqrt(var) / ((double) i);
	}

	// check if any of the moving averages is better than the reference (by at least tolerance relative improvement)
	double ref = moving_avg[0];
	bool improved = false;
	for (int i = 1; i < moving_avg.size(); ++i) {
		//      fprintf(stderr,"ref=%g tol=%g moving=%g i=%d moving_avgavg=%g error=%g\n",ref,tolerance,moving_avg[i],i,moving_avgavg,error); fflush(stderr);
		if (moreIsBetter)
			improved |= (moving_avg[i] > ref * (1.0 + tolerance));
		else
			improved |= (moving_avg[i] < ref * (1.0 - tolerance));
	}

	// estimate normalized jump for controlling tolerance as approach stopping point
	if (moving_avg.size() >= 2) {
		//      *jump = (*std::max_element(moving_avg.begin(), moving_avg.end()) - *std::min_element(moving_avg.begin(), moving_avg.end()))/(DBL_EPSILON+norm);
		*jump = (moving_avg.front() - moving_avg.back())
				/ (DBL_EPSILON + moving_avg.front() + moving_avg.back());
	} else {
		*jump = DBL_MAX;
	}

	if (improved) {
		if (improved && verbose)
			cout << "improved from " << ref << " to "
					<< (moreIsBetter ?
							*std::max_element(moving_avg.begin(),
									moving_avg.end()) :
							*std::min_element(moving_avg.begin(),
									moving_avg.end())) << endl;
		return false;
	} else {
		if (verbose)
			cout << "stopped." << endl;
		return true;
	}
}


// Function: fileExists
/**
    Check if a file exists
@param[in] filename - the name of the file to check

@return    true if the file exists, else false

*/
bool fileExists(const std::string& filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}
// Elastic Net
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda \alpha ||x||_1 + \lambda 1-\alpha ||x||_2
//
// for many values of \lambda and multiple values of \alpha
// See <h2o4gpu>/matlab/examples/lasso_path.m for detailed description.
// m and n are training data size
#define TRAINError 0
#define CVError 1
#define VALIDError 2

#define NUMError 3 // train, hold-out CV, valid
#define NUMOTHER 3 // for lambda, alpha, tol

template<typename T>
double ElasticNetptr(
		const char family, int dopredict, int sourceDev, int datatype, int sharedA,
		int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n,
		size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		T *alphas, T *lambdas,
		double tol, double tolseekfactor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction, int max_iterations, int verbose,
		void *trainXptr, void *trainYptr, void *validXptr, void *validYptr,
		void *weightptr, int givefullpath, T **Xvsalphalambda, T **Xvsalpha,
		T **validPredsvsalphalambda, T **validPredsvsalpha, size_t *countfull,
		size_t *countshort, size_t *countmore) {

	if(0){ // DEBUG
	if(alphas!=NULL){
	for(int i=0;i<nAlphas;i++){
		fprintf(stderr,"alpha[%d]=%g",i,alphas[i]);
	}
	}
	if(lambdas!=NULL){
	for(int i=0;i<nLambdas;i++){
		fprintf(stderr,"lambdas[%d]=%g",i,lambdas[i]);
	}
	}
	fprintf(stderr,"lambda_min_ratio=%g", lambda_min_ratio);
	fprintf(stderr,"lambda_max=%g", lambda_max);
	fflush(stderr);
	}

	if (dopredict == 0) {
		return ElasticNetptr_fit(family, sourceDev, datatype, sharedA, nThreads, gpu_id, nGPUs, totalnGPUs,
				ord, mTrain, n, mValid, intercept, standardize,
				lambda_max, lambda_min_ratio, nLambdas, nFolds,
				nAlphas, alpha_min, alpha_max,
				alphas, lambdas,
				tol, tolseekfactor,
				lambdastopearly, glmstopearly, stopearlyerrorfraction, max_iterations, verbose, trainXptr,
				trainYptr, validXptr, validYptr, weightptr, givefullpath,
				Xvsalphalambda, Xvsalpha, validPredsvsalphalambda,
				validPredsvsalpha, countfull, countshort, countmore);
	} else {
		return ElasticNetptr_predict(family, sourceDev, datatype, sharedA, nThreads, gpu_id, nGPUs, totalnGPUs,
				ord, mTrain, n, mValid, intercept, standardize,
				lambda_max, lambda_min_ratio, nLambdas, nFolds,
				nAlphas, alpha_min, alpha_max,
				alphas, lambdas,
				tol, tolseekfactor,
				lambdastopearly, glmstopearly, stopearlyerrorfraction, max_iterations, verbose, trainXptr,
				trainYptr, validXptr, validYptr, weightptr, givefullpath,
				Xvsalphalambda, Xvsalpha, validPredsvsalphalambda,
				validPredsvsalpha, countfull, countshort, countmore);
	}

}

#define MAPXALL(i,a,which) (which + a*(n+NUMError+NUMOTHER) + i*(n+NUMError+NUMOTHER)*nAlphas)
#define MAPXBEST(a,which) (which + a*(n+NUMError+NUMOTHER))

#define MAPPREDALL(i,a,which, m) (which + a*m + i*m*nAlphas)
#define MAPPREDBEST(a,which, m) (which + a*m)

template<typename T>
double ElasticNetptr_fit(const char family, int sourceDev, int datatype, int sharedA, int nThreads,
		int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n, size_t mValid,
		int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		T *alphas, T *lambdas,
		double tol, double tolseekfactor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction,
		int max_iterations, int verbose, void *trainXptr, void *trainYptr,
		void *validXptr, void *validYptr, void *weightptr, int givefullpath,
		T **Xvsalphalambda, T **Xvsalpha, T **validPredsvsalphalambda,
		T **validPredsvsalpha, size_t *countfull, size_t *countshort,
		size_t *countmore) {

	if (0) {
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(-100, 100);
		//T arr[] = {3,6,2,8,10,54,5,9};
		int myn = 100;
		T arr[myn];
		for (int ii = 0; ii < myn; ii++) {
			arr[ii] = distribution(generator);
			fprintf(stderr, "%g ", arr[ii]);
			fflush(stderr);
		}
		fprintf(stderr, "\n");
		fflush(stderr);
		int whichbeta[NUMBETA];
		T valuebeta[NUMBETA];
		int myk = 5;
		int whichmax = 1; // 0 : larger  1: largest absolute magnitude
		h2o4gpu::topkwrap(1, myn, myk, arr, &whichbeta[0], &valuebeta[0]);
		for (int ii = 0; ii < myk; ii++) {
			fprintf(stderr, "%d %g ", whichbeta[ii], valuebeta[ii]);
			fflush(stderr);
		}
		fprintf(stderr, "\n");
		fflush(stderr);
		exit(0);
	}


	// Adjust any parameters for user friendliness
	nAlphas = std::max(nAlphas,0); // At least zero alphas
	nLambdas = std::max(nLambdas,0); // At least zero Lambdas



	signal(SIGINT, my_function);
	signal(SIGTERM, my_function);
	int nlambda = nLambdas;
	if (VERBOSEENET || verbose>3) {
		cout << "Hardware: " << HARDWARE << endl;
	}

	// number of openmp threads = number of cuda devices to use
#ifdef _OPENMP
	int omt=omp_get_max_threads();
	//      omp_set_num_threads(MIN(omt,nGPUs));  // not necessary, but most useful mode so far
	omp_set_num_threads(nThreads);// not necessary, but most useful mode so far
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
	size_t realfolds = (nFolds == 0 ? 1 : nFolds);
	size_t totalfolds = nFolds * (nFolds > 1 ? 2 : 1);
	DEBUG_FPRINTF(stderr, "Set folds=%d realfolds=%zu Total Folds=%zu\n",
			nFolds, realfolds, totalfolds);

	if (VERBOSEENET || verbose>3) {
		fprintf(stderr, "Before malloc X\n");
		fflush(stderr);
	}
	// setup storage for returning results back to user
	// iterate over predictors (n) or other information fastest so can memcpy X
	*countmore = NUMError + NUMOTHER;
	if (givefullpath) {
		*countfull = nLambdas * nAlphas * (n + *countmore);
		*Xvsalphalambda = (T*) calloc(*countfull, sizeof(T)); // +NUMOTHER for values of lambda, alpha, and tolerance
	} else { // only give back solution for optimal lambda after CV is done
		*countfull = 0;
		*Xvsalphalambda = NULL;
	}
	*countshort = nAlphas * (n + *countmore);
	*Xvsalpha = (T*) calloc(*countshort, sizeof(T));
	//    printf("inside: countfull=%zu countshort=%zu countmore=%zu\n",*countfull,*countshort,*countmore); fflush(stdout);
	if (VERBOSEENET || verbose>3) {
		fprintf(stderr, "After malloc X\n");
		fflush(stderr);
	}

	// for source, create class objects that creates cuda memory, cpu memory, etc.
	// This takes-in raw GPU pointer
	//  h2o4gpu::MatrixDense<T> Asource_(sourceDev, ord, mTrain, n, mValid, reinterpret_cast<T *>(trainXptr));
	// assume source thread is 0th thread (TODO: need to ensure?)
	if (VERBOSEENET || verbose>3) {
		fprintf(stderr, "Before Asource\n");
		fflush(stderr);
	}
	int sourceme = sourceDev;
	h2o4gpu::MatrixDense<T> Asource_(sharedA, sourceme, sourceDev, datatype,
			ord, mTrain, n, mValid, reinterpret_cast<T *>(trainXptr),
			reinterpret_cast<T *>(trainYptr), reinterpret_cast<T *>(validXptr),
			reinterpret_cast<T *>(validYptr), reinterpret_cast<T *>(weightptr));
	if (VERBOSEENET || verbose>3) {
		fprintf(stderr, "After Asource\n");
		fflush(stderr);
	}
	// now can always access A_(sourceDev) to get pointer from within other MatrixDense calls
	T min[2], max[2], mean[2], var[2], sd[2], skew[2], kurt[2];
	T lambdamax0;
	Asource_.Stats(intercept, min, max, mean, var, sd, skew, kurt, lambdamax0);
	double sdTrainY = (double) sd[0], meanTrainY = (double) mean[0];
	double sdValidY = (double) sd[1], meanValidY = (double) mean[1];
	if(lambda_max<0.0){ // set if user didn't set
		lambda_max = (double) lambdamax0;
	}else if(lambda_max >= 0.0){

	}else{
		cerr << "Invalid lambda_max " << lambda_max << endl;
		exit(0);
	}
//	fprintf(stderr,"lambda_max=%g lambdamax0=%g\n",lambda_max,lambdamax0); fflush(stderr);
	if (VERBOSEENET || verbose>3) {
		fprintf(stderr, "After stats\n");
		fflush(stderr);
	}

	if (verbose || VERBOSEANIM || VERBOSEENET) {

		FILE *myfile;
		for (size_t filei = 0; filei <= 2; filei++) {
			if (filei == 0 && verbose>2)
				myfile = stderr;
			else if (filei == 1 && verbose>2)
				myfile = stdout;
			else if(verbose>2) {
				myfile = fopen("stats.txt", "wt");
				if (myfile == NULL)
					continue; // skip if can't write, don't complain
			}
			else myfile = NULL;
		}
		if(myfile!=NULL){
			fprintf(myfile, "min");
			for (int ii = 0; ii <= (mValid > 0 ? 1 : 0); ii++)
				fprintf(myfile, " %21.15g", min[ii]);
			fprintf(myfile, "\n");
			fprintf(myfile, "max");
			for (int ii = 0; ii <= (mValid > 0 ? 1 : 0); ii++)
				fprintf(myfile, " %21.15g", max[ii]);
			fprintf(myfile, "\n");
			fprintf(myfile, "mean");
			for (int ii = 0; ii <= (mValid > 0 ? 1 : 0); ii++)
				fprintf(myfile, " %21.15g", mean[ii]);
			fprintf(myfile, "\n");
			fprintf(myfile, "var");
			for (int ii = 0; ii <= (mValid > 1 ? 1 : 0); ii++)
				fprintf(myfile, " %21.15g", var[ii]);
			fprintf(myfile, "\n");
			fprintf(myfile, "sd");
			for (int ii = 0; ii <= (mValid > 0 ? 1 : 0); ii++)
				fprintf(myfile, " %21.15g", sd[ii]);
			fprintf(myfile, "\n");
			fprintf(myfile, "skew");
			for (int ii = 0; ii <= (mValid > 1 ? 1 : 0); ii++)
				fprintf(myfile, " %21.15g", skew[ii]);
			fprintf(myfile, "\n");
			fprintf(myfile, "kurt");
			for (int ii = 0; ii <= (mValid > 1 ? 1 : 0); ii++)
				fprintf(myfile, " %21.15g", kurt[ii]);
			fprintf(myfile, "\n");
			fprintf(myfile, "lambda_max=%g\n", lambda_max);
			fflush(myfile);
		}
	}

	// temporarily get trainX, etc. from h2o4gpu (which may be on gpu)
	T *trainX = NULL;
	T *trainY = NULL;
	T *validX = NULL;
	T *validY = NULL;
	T *trainW = NULL;
	if (OLDPRED)
		trainX = (T *) malloc(sizeof(T) * mTrain * n);
	trainY = (T *) malloc(sizeof(T) * mTrain);
	if (OLDPRED)
		validX = (T *) malloc(sizeof(T) * mValid * n);
	validY = (T *) malloc(sizeof(T) * mValid);
	trainW = (T *) malloc(sizeof(T) * mTrain);

	if (OLDPRED)
		Asource_.GetTrainX(datatype, mTrain * n, &trainX);
	Asource_.GetTrainY(datatype, mTrain, &trainY);
	if (OLDPRED)
		Asource_.GetValidX(datatype, mValid * n, &validX);
	Asource_.GetValidY(datatype, mValid, &validY);
	Asource_.GetWeight(datatype, mTrain, &trainW);

	T alphaarray[realfolds * 2][nAlphas]; // shared memory space for storing alpha for various folds and alphas
	T lambdaarray[realfolds * 2][nAlphas]; // shared memory space for storing lambda for various folds and alphas
	T tolarray[realfolds * 2][nAlphas]; // shared memory space for storing tolerance for various folds and alphas
	// which error to use for final check of which model is best (keep validation fractional data for purely reporting)
	int owhicherror;
	if (mValid > 0) {
		if (realfolds <= 1)
			owhicherror = 2;
		else
			owhicherror = 2;
	} else {
		if (realfolds <= 1)
			owhicherror = 0;
		else
			owhicherror = 1;
	}
	// which error to use within lambda-loop to decide if accurate model
	int iwhicherror;
	if (mValid > 0) {
		if (realfolds <= 1)
			iwhicherror = 2;
		else
			iwhicherror = 1;
	} else {
		if (realfolds <= 1)
			iwhicherror = 0;
		else
			iwhicherror = 1;
	}
#define ErrorLOOP(ri) for(int ri=0;ri<NUMError;ri++)
	T errorarray[NUMError][realfolds * 2][nAlphas]; // shared memory space for storing error for various folds and alphas
#define MAX(a,b) ((a)>(b) ? (a) : (b))
	// Setup each thread's h2o4gpu
	double t = timer<double>();
	double t1me0;

	int verboseanimtriggered=0;
	FILE *filerror = NULL;
	FILE *filvarimp = NULL;

	// critical files for all threads
	char filename0[100];
	sprintf(filename0, "error.txt");
	if(fileExists(filename0)){ // if at least 0 size file, then assume created for animation purposes
		fprintf(stderr,"Doing Animation Output\n"); fflush(stderr);
		verboseanimtriggered=1;
		filerror = fopen(filename0, "wt");
		if (filerror == NULL) {
			cerr << "Cannot open filename0=" << filename0 << endl;
			exit(0);
		}
		sprintf(filename0, "varimp.txt");
		filvarimp = fopen(filename0, "wt");
		if (filvarimp == NULL) {
			cerr << "Cannot open filename0=" << filename0 << endl;
			exit(0);
		}
	}

	////////////////////////////////
	// PARALLEL REGION
#if USEPARALLEL != 0
#pragma omp parallel proc_bind(master)
#endif
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
		int me = 0;
#endif

		int blasnumber;
#ifdef HAVECUDA
		blasnumber=CUDA_MAJOR;
#else
		blasnumber = mklperthread; // roughly accurate for openblas as well
#endif

		// choose GPU device ID for each thread
		int wDev = gpu_id + (nGPUs > 0 ? me % nGPUs : 0);
		wDev = wDev % totalnGPUs;

		FILE *fil = NULL;
		if(VERBOSEANIM){
		// Setup file output
		char filename[100];
		sprintf(filename, "me%d.%d.%s.%s.%d.%d.%d.txt", me, wDev, _GITHASH_,
				TEXTARCH, sharedA, nThreads, nGPUs);
		fil = fopen(filename, "wt");
		if (fil == NULL) {
			cerr << "Cannot open filename=" << filename << endl;
			exit(0);
		} else
			fflush(fil);
		}

		////////////
		//
		// create class objects that creates cuda memory, cpu memory, etc.
		//
		////////////
		double t0 = timer<double>();
		DEBUG_FPRINTF(fil, "Moving data to the GPU. Starting at %21.15g\n", t0);
#pragma omp barrier // not required barrier
		h2o4gpu::MatrixDense<T> A_(sharedA, me, wDev, Asource_);
#pragma omp barrier // required barrier for wDev=sourceDev so that Asource_._data (etc.) is not overwritten inside h2o4gpu_data(wDev=sourceDev) below before other cores copy data
		h2o4gpu::H2O4GPUDirect<T, h2o4gpu::MatrixDense<T> > h2o4gpu_data(
				sharedA, me, wDev, A_);
#pragma omp barrier // not required barrier
		double t1 = timer<double>();
		if (me == 0) { //only thread=0 times entire post-warmup procedure
			t1me0 = t1;
		}
		DEBUG_FPRINTF(fil, "Done moving data to the GPU. Stopping at %21.15g\n",
				t1);
		DEBUG_FPRINTF(fil, "Done moving data to the GPU. Took %g secs\n",
				t1 - t0);

		///////////////////////////////////////////////////
		// BEGIN SVD
		if (0) {
			A_.svd1();
		}

		////////////////////////////////////////////////
		// BEGIN GLM

		// Setup constant parameters for all models
		h2o4gpu_data.SetnDev(1); // set how many cuda devices to use internally in h2o4gpu
		//    h2o4gpu_data.SetRelTol(1e-4); // set how many cuda devices to use internally in h2o4gpu
		//    h2o4gpu_data.SetAbsTol(1e-5); // set how many cuda devices to use internally in h2o4gpu
		//    h2o4gpu_data.SetAdaptiveRho(true);
		//h2o4gpu_data.SetEquil(false);
		//      h2o4gpu_data.SetRho(1E-6);
		//      h2o4gpu_data.SetRho(1E-3);
		h2o4gpu_data.SetRho(1.0);
		h2o4gpu_data.SetVerbose(verbose);
		h2o4gpu_data.SetStopEarly(glmstopearly);
		h2o4gpu_data.SetStopEarlyErrorFraction(stopearlyerrorfraction);
		h2o4gpu_data.SetMaxIter(max_iterations);

		DEBUG_FPRINTF(fil, "BEGIN SOLVE: %d\n", 0);
		int fi, a;

		T *X0 = new T[n]();
		T *L0 = new T[mTrain]();
		int gotpreviousX0 = 0;

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
		double errorarrayofa[NUMError][nAlphas];
		for (int lambdatype = 0; lambdatype <= (realfolds > 1); lambdatype++) {
			size_t nlambdalocal;

			// Set Lambda
			std::vector<T> lambdaslocal(nlambda);
			if (lambdatype == LAMBDATYPEPATH) {
				nlambdalocal = nlambda;
				const T lambda_min = lambda_min_ratio
						* static_cast<T>(lambda_max); // like h2o4gpu.R
				T lambda_max_use = lambda_max; // std::max(static_cast<T>(1e-2), alpha); // same as H2O
				DEBUG_FPRINTF(stderr, "lambda_max: %f\n", lambda_max_use);
				DEBUG_FPRINTF(stderr, "lambda_min: %f\n", lambda_min);
				DEBUG_FPRINTF(fil, "lambda_max: %f\n", lambda_max_use);
				DEBUG_FPRINTF(fil, "lambda_min: %f\n", lambda_min);
				// Regularization path: geometric series from lambda_max_use to lambda_min
				if(lambdas==NULL){
					if (nlambdalocal > 1) {
						double dec = std::pow(lambda_min_ratio,
								1.0 / (nlambdalocal - 1.));
						lambdaslocal[0] = lambda_max_use;
						for (int i = 1; i < nlambdalocal; ++i)
							lambdaslocal[i] = lambdaslocal[i - 1] * dec;
					} else { // use minimum, so user can control the value of lambda used
						lambdaslocal[0] = lambda_min_ratio * lambda_max_use;
					}
				}
				else{
					for (int i = 1; i < nlambdalocal; ++i){
						lambdaslocal[i] = lambdas[i];
					}
				}
			} else {
				nlambdalocal = 1;
			}

			//////////////////////////////
			//
			// LOOP OVER FOLDS AND ALPHAS
			//
			///////////////////////////////
#pragma omp for schedule(dynamic,1) collapse(2)
			for (a = 0; a < nAlphas; ++a) { //alpha search
				for (fi = 0; fi < realfolds; ++fi) { //fold

					////////////
					// SETUP ALPHA
					T alpha;
					if(alphas==NULL){
						if(nAlphas<=1){
							alpha = (alpha_min + alpha_max)*0.5;
						}
						else{
							alpha = alpha_min + (alpha_max - alpha_min) * static_cast<T>(a) / static_cast<T>(nAlphas - 1);
						}
					}
					else{
						alpha = alphas[a];
					}
					// Setup Lambda in case not doing lambda path
					if (lambdatype == LAMBDATYPEONE){
						lambdaslocal[0] = lambdaarrayofa[a];
					}

					/////////////
					//
					// SETUP FOLD (and weights)
					//
					////////////
					// FOLDTYPE: 0 = any portion and can be overlapping
					// FOLDTYPE: 1 = non-overlapping folds
#define FOLDTYPE 1
					T fractrain;
					if (FOLDTYPE == 0) {
						fractrain = (realfolds > 1 ? 0.8 : 1.0);
					} else {
						fractrain = (
								realfolds > 1 ?
										1.0 - 1.0 / ((double) realfolds) : 1.0);
					}
					T fracvalid = 1.0 - fractrain;
					T weights[mTrain];
					if (realfolds > 1) {
						for (unsigned int j = 0; j < mTrain; ++j) {
							T foldon = 1;
							int jfold = j - fi * fracvalid * mTrain;
							if (jfold >= 0 && jfold < fracvalid * mTrain)
								foldon = 1E-13;

							weights[j] = foldon * trainW[j];
							//              fprintf(stderr,"a=%d fold=%d j=%d foldon=%g trainW=%g weights=%g\n",a,fi,j,foldon,trainW[j],weights[j]); fflush(stderr);
						}
					} else {   // then assume meant one should just copy weights
						for (unsigned int j = 0; j < mTrain; ++j)
							weights[j] = trainW[j];
					}

					// normalize weights before input (method has issue with small typical weights, so avoid normalization and just normalize in error itself only)
					T sumweight = 0, maxweight = -std::numeric_limits<T>::max();
					for (unsigned int j = 0; j < mTrain; ++j)
						sumweight += weights[j];
					for (unsigned int j = 0; j < mTrain; ++j) {
						if (maxweight < weights[j])
							maxweight = weights[j];
					}
					if (0) {
						if (sumweight != 0.0) {
							for (unsigned int j = 0; j < mTrain; ++j)
								weights[j] /= sumweight;
						} else
							continue; // skip this fi,a
						//            fprintf(stderr,"a=%d fold=%d sumweights=%g\n",a,fi,sumweight); fflush(stderr);
					}

					////////////////////////////
					//
					// LOOP OVER LAMBDA
					//
					///////////////////////////////
					vector<double> scoring_history;
					int gotX0 = 0;
					double jump = DBL_MAX;
					double norm = (mValid == 0 ? sdTrainY : sdValidY);
					int skiplambdaamount = 0;
					int i;
					double trainError = -1;
					double ivalidError = -1;
					double validError = -1;
					//double tol = 1E-2; // highest acceptable tolerance (USER parameter)  Too high and won't go below standard deviation.
					double tolnew = tol;
					T lambda = -1;
					double tbestalpha = -1, tbestlambda = -1, tbesttol =
							std::numeric_limits<double>::max(),
							tbesterror[NUMError];
					ErrorLOOP(ri)
						tbesterror[ri] = std::numeric_limits<double>::max();


					////////////////////////////////
					// LOOP over lambda
					for (i = 0; i < nlambdalocal; ++i) {
						if (flag) {
							continue;
						}

						// Set Lambda
						lambda = lambdaslocal[i];
						DEBUG_FPRINTF(fil, "lambda %d = %f\n", i, lambda);

						//////////////
						//
						// if lambda path, control how go along path
						//
						//////////////
						if (lambdatype == LAMBDATYPEPATH) {

							// Reset Solution if starting fresh for this alpha
							if (i == 0) {
								// see if have previous solution for new alpha for better warmstart
								if (gotpreviousX0) {
									//              DEBUG_FPRINTF(stderr,"m=%d a=%d i=%d Using old alpha solution\n",me,a,i);
									//              for(unsigned int ll=0;ll<n;ll++) DEBUG_FPRINTF(stderr,"X0[%d]=%g\n",ll,X0[ll]);
									h2o4gpu_data.SetInitX(X0);
									h2o4gpu_data.SetInitLambda(L0);
								} else {
									h2o4gpu_data.ResetX(); // reset X if new alpha if expect much different solution
								}
							}

							///////////////////////
							//
							// Set tolerances more automatically
							// (NOTE: that this competes with stopEarly() below in a good way so that it doesn't stop overly early just because errors are flat due to poor tolerance).
							// Note currently using jump or jumpuse.  Only using scoring vs. standard deviation.
							// To check total iteration count, e.g., : grep -a "Iter  :" output.txt|sort -nk 3|awk '{print $3}' | paste -sd+ | bc
							double jumpuse = DBL_MAX;
							//h2o4gpu_data.SetRho(maxweight); // can't trust warm start for rho, because if adaptive rho is working hard to get primary or dual residuals below eps, can drive rho out of control even though residuals and objective don't change in error, but then wouldn't be good to start with that rho and won't find solution for any other latter lambda or alpha.  Use maxweight to scale rho, because weight and lambda should scale the same way.
							tolnew = tol; //*lambda/lambdaslocal[0]; // as lambda gets smaller, so must attempt at relative tolerance, in order to capture affect of lambda regularization on primary term (that is otherwise order unity unless weights are not unity).
							h2o4gpu_data.SetRelTol(tolnew);
							h2o4gpu_data.SetAbsTol(
									1.0 * std::numeric_limits<T>::epsilon()); // way code written, has 1+rho and other things where catastrophic cancellation occur for very small weights or rho, so can't go below certain absolute tolerance.  This affects adaptive rho and how warm-start on rho would work.
											// see if getting below stddev, if so decrease tolerance
							if (scoring_history.size() >= 1) {
								double ratio = (norm - scoring_history.back())
										/ norm;

								if (ratio > 0.0) {
									double factor = 0.05; // rate factor (USER parameter)
									double tollow = tolseekfactor * tol; //*lambda/lambdaslocal[0]; //lowest allowed tolerance (USER parameter)
									tolnew = tol * pow(2.0, -ratio / factor); //*lambda/lambdaslocal[0]
									if (tolnew < tollow)
										tolnew = tollow;

									h2o4gpu_data.SetRelTol(tolnew);
									h2o4gpu_data.SetAbsTol(
											1.0
													* std::numeric_limits<T>::epsilon()); // way code written, has 1+rho and other things where catastrophic cancellation occur for very small weights or rho, so can't go below certain absolute tolerance.
									jumpuse = jump;
								}
								//              fprintf(stderr,"me=%d a=%d i=%d jump=%g jumpuse=%g ratio=%g tolnew=%g norm=%g score=%g\n",me,a,i,jump,jumpuse,ratio,tolnew,norm,scoring_history.back());
							}
						} else { // single lambda
								 // assume warm-start value of X and other internal variables
								 //                fprintf(stderr,"tolnew to use for last alpha=%g lambda=%g is %g\n",alphaarrayofa[a],lambdaarrayofa[a],tolarrayofa[a]); fflush(stderr);
							tolnew = tolarrayofa[a];
							h2o4gpu_data.SetRelTol(tolnew);
							h2o4gpu_data.SetAbsTol(
									10.0 * std::numeric_limits<T>::epsilon()); // way code written, has 1+rho and other things where catastrophic cancellation occur for very small weights or rho, so can't go below certain absolute tolerance.
						}

						////////////////////
						//
						// Solve
						//
						////////////////////
						// setup f,g as functions of alpha
						std::vector<FunctionObj<T>> f;
						std::vector<FunctionObj<T>> g;
						f.reserve(mTrain);
						g.reserve(n);

						/*
						Start logic for type of `family` argument passed in
						*/
						if(family == 'e'){ //elasticnet
							// minimize ||Ax-b||_2^2 + \alpha\lambda||x||_1 + (1/2)(1-alpha)*lambda x^2
							for (unsigned int j = 0; j < mTrain; ++j) f.emplace_back(kSquare, 1.0, trainY[j], weights[j]); // h2o4gpu.R
							for (unsigned int j = 0; j < n - intercept; ++j) g.emplace_back(kAbs);
							if (intercept) g.emplace_back(kZero);
						}else if(family == 'l'){ //logistic
							// minimize \sum_i -d_i y_i + log(1 + e ^ y_i) + \lambda ||x||_1
							for (unsigned int j = 0; j < mTrain; ++j) f.emplace_back(kLogistic, 1.0, 0.0, weights[j], -weights[j]*trainY[j]); // h2o4gpu.R
							for (unsigned int j = 0; j < n - intercept; ++j) g.emplace_back(kAbs);
							if (intercept) g.emplace_back(kZero);
						// }else if(family == 's'){ //svm
						// 	// minimize (1/2) ||w||_2^2 + \lambda \sum (a_i^T * [w; b] + 1)_+.
						// 	for (unsigned int j = 0; j < mTrain; ++j) f.emplace_back(kMaxPos0, 1.0, -1.0, weights[j]*lambda); // h2o4gpu.R}
						// 	for (unsigned int j = 0; j < n - intercept; ++j) g.emplace_back(kSquare);
						// 	if (intercept) g.emplace_back(kZero);
						}else{
							//throw error
							throw "Wrong family type selected. Should be either elasticnet or logistic";
						}
						T penalty_factor = static_cast<T>(1.0); // like h2o4gpu.R
						// assign lambda (no penalty for intercept, the last coeff, if present)
						for (unsigned int j = 0; j < n - intercept; ++j) {
							g[j].c = static_cast<T>(alpha * lambda
									* penalty_factor); //for L1
							g[j].e = static_cast<T>((1.0 - alpha) * lambda
									* penalty_factor); //for L2
						}
						if (intercept) {
							g[n - 1].c = 0;
							g[n - 1].e = 0;
						}
						// Solve
						h2o4gpu_data.Solve(f, g);

						int doskiplambda = 0;
						if (lambdatype == LAMBDATYPEPATH) {
							/////////////////
							//
							// Check if getting solution was too easy and was 0 iterations.  If so, overhead is not worth it, so try skipping by 1.
							//
							/////////////////
							if (h2o4gpu_data.GetFinalIter() == 0) {
								doskiplambda = 1;
								skiplambdaamount++;
							} else {
								// reset if not 0 iterations
								skiplambdaamount = 0;
							}

							////////////////////////////////////////////
							//
							// Check if solution was found
							//
							////////////////////////////////////////////
							int maxedout = 0;
							if (h2o4gpu_data.GetFinalIter()
									== h2o4gpu_data.GetMaxIter())
								maxedout = 1;
							else
								maxedout = 0;

							if (maxedout)
								h2o4gpu_data.ResetX(); // reset X if bad solution so don't start next lambda with bad solution
							// store good high-lambda solution to start next alpha with (better than starting with low-lambda solution)
							if (gotX0 == 0 && maxedout == 0) {
								gotX0 = 1;
								// TODO: FIXME: Need to get (and have solver set) best solution or return all, because last is not best.
								gotpreviousX0 = 1;
								memcpy(X0, &h2o4gpu_data.GetX()[0],
										n * sizeof(T));
								memcpy(L0, &h2o4gpu_data.GetLambda()[0],
										mTrain * sizeof(T));
							}

						}

						if (intercept) {
							DEBUG_FPRINTF(fil, "intercept: %g\n",
									h2o4gpu_data.GetX()[n - 1]);
							DEBUG_FPRINTF(stdout, "intercept: %g\n",
									h2o4gpu_data.GetX()[n - 1]);
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
								if (std::abs(h2o4gpu_data.GetX()[i]) > 1e-8) {
									dof++;
								}
							}
						}

						int whichbeta[NUMBETA];
						T valuebeta[NUMBETA];
						int whichmax = 1; // 0 : larger  1: largest absolute magnitude
						h2o4gpu::topkwrap(whichmax, (int) (n - intercept),
								(int) (NUMBETA),
								const_cast<T*>(&h2o4gpu_data.GetX()[0]),
								&whichbeta[0], &valuebeta[0]);

						//              memcpy(X0,&h2o4gpu_data.GetX()[0],n*sizeof(T));
						if (0) {
							std::sort(const_cast<T*>(&h2o4gpu_data.GetX()[0]),
									const_cast<T*>(&h2o4gpu_data.GetX()[n
											- intercept]));
							for (size_t i = 0; i < n - intercept; ++i) {
								fprintf(stderr, "BETA: i=%zu beta=%g\n", i,
										h2o4gpu_data.GetX()[i]);
								fflush(stderr);
							}
						}

						// TRAIN PREDS
#if(OLDPRED)
						std::vector <T> trainPreds(mTrain);
						for (size_t i = 0; i < mTrain; ++i) {
							trainPreds[i] = 0;
							for (size_t j = 0; j < n; ++j) {
								trainPreds[i] += h2o4gpu_data.GetX()[j] * trainX[i * n + j]; //add predictions
							}
						}
#else
						std::vector<T> trainPreds(
								&h2o4gpu_data.GettrainPreds()[0],
								&h2o4gpu_data.GettrainPreds()[0] + mTrain);
						//              for(unsigned int iii=0;iii<mTrain;iii++){
						//                fprintf(stderr,"trainPreds[%d]=%g\n",iii,trainPreds[iii]);
						//              }
#endif
						//Compute inverse logit of predictions if family == 'logistic' to get actual probabilities.
						if(family == 'l'){
							std::transform(trainPreds.begin(), trainPreds.end(), trainPreds.begin(),[](T i) -> T { return 1/(1+exp(-i)); });
						}
						// Error: TRAIN
						trainError = h2o4gpu::getError(weights, mTrain,
								&trainPreds[0], trainY, family);

						if(verbose){
							if(family == 'l'){
								std::cout << "Training Logloss = " << trainError << " for lambda = " << lambda << " and alpha = " << alpha << std::endl;
							} else {
								std::cout << "Training RMSE = " << trainError << " for lambda = " << lambda << " and alpha = " << alpha << std::endl;
							}
						}
						if (standardize) {
							trainError *= sdTrainY;
							for (size_t i = 0; i < mTrain; ++i) {
								// reverse standardization
								trainPreds[i] *= sdTrainY; //scale
								trainPreds[i] += meanTrainY; //intercept
								//assert(trainPreds[i] == h2o4gpu_data.GetY()[i]); //FIXME: CHECK
							}
						}

						// Error: on fold's held-out training data
						if (realfolds > 1) {
							const T offset = 1.0;
							ivalidError = h2o4gpu::getError(offset, weights,
									mTrain, &trainPreds[0], trainY, family);
							if(verbose){
								if(family == 'l'){
									std::cout << "Average CV Logloss = " << ivalidError << " for lambda = " << lambda << " and alpha = " << alpha << std::endl;
								} else {
									std::cout << "Average CV RMSE = " << ivalidError << " for lambda = " << lambda << " and alpha = " << alpha << std::endl;
								}
							}
						} else {
							ivalidError = -1.0;
						}

						// VALID (preds and error)
						validError = -1;
						if (mValid > 0) {

							T weightsvalid[mValid];
							for (size_t i = 0; i < mValid; ++i) {          //row
								weightsvalid[i] = 1.0;
							}

							// Valid Preds
#if(OLDPRED)
							std::vector <T> validPreds(mValid);
							for (size_t i = 0; i < mValid; ++i) { //row
								validPreds[i] = 0;
								for (size_t j = 0; j < n; ++j) { //col
									validPreds[i] += h2o4gpu_data.GetX()[j] * validX[i * n + j];//add predictions
								}
							}
#else

							std::vector<T> validPreds(
									&h2o4gpu_data.GetvalidPreds()[0],
									&h2o4gpu_data.GetvalidPreds()[0] + mValid);
#endif
							//Compute inverse logit of predictions if family == 'logistic' to get actual probabilities.
							if(family == 'l'){
								std::transform(validPreds.begin(), validPreds.end(), validPreds.begin(),[](T i) -> T { return 1/(1+exp(-i)); });
							}
							// Error: VALIDs
							validError = h2o4gpu::getError(weightsvalid, mValid,
									&validPreds[0], validY, family);

							if(verbose){
								if(family == 'l'){
									std::cout << "Validation Logloss = " << validError << " for lambda = " << lambda << " and alpha = " << alpha << std::endl;
								} else {
									std::cout << "Validation RMSE = " << validError << " for lambda = " << lambda << " and alpha = " << alpha << std::endl;
								}
							}

							if (standardize) {
								validError *= sdTrainY;
								for (size_t i = 0; i < mValid; ++i) { //row
									// reverse (fitted) standardization
									validPreds[i] *= sdTrainY; //scale
									validPreds[i] += meanTrainY; //intercept
								}
							}
						}

						////////////
						//
						// report scores
						//
						////////////
						if (VERBOSEENET)
							Printmescore(fil);
#pragma omp critical
						{
							if (VERBOSEENET)
								Printmescore(stdout);
							if (VERBOSEANIM || verboseanimtriggered==1){
								Printmescoresimple(filerror);
								Printmescoresimple2(filvarimp);
							}
						}

						T localerror[NUMError];
						localerror[0] = trainError;
						localerror[1] = ivalidError;
						localerror[2] = validError;
						if (tbesterror[iwhicherror] > localerror[iwhicherror]) {
							tbestalpha = alpha;
							tbestlambda = lambda;
							tbesttol = tolnew;
							ErrorLOOP(ri)
								tbesterror[ri] = localerror[ri];
						}

						// save scores
						scoring_history.push_back(localerror[iwhicherror]);

						if (lambdatype == LAMBDATYPEPATH) {
							if (fi == 0 && givefullpath) { // only store first fold for user
								//#define MAPXALL(i,a,which) (which + a*(n+NUMError+NUMOTHER) + i*(n+NUMError+NUMOTHER)*nlambdas)
								//#define MAPXBEST(a,which) (which + a*(n+NUMError+NUMOTHER))
								//#define NUMOTHER 3 // for lambda, alpha, tolnew
								// Save solution to return to user
								memcpy(&((*Xvsalphalambda)[MAPXALL(i, a, 0)]),
										&h2o4gpu_data.GetX()[0],
										n * sizeof(T));
								// Save error to return to user
								ErrorLOOP(ri)
									(*Xvsalphalambda)[MAPXALL(i, a, n + ri)] =
											localerror[ri];
								// Save lambda to return to user
								(*Xvsalphalambda)[MAPXALL(i, a, n+NUMError)] =
										lambda;
								// Save alpha to return to user
								(*Xvsalphalambda)[MAPXALL(i, a, n+NUMError+1)] =
										alpha;
								// Save tolnew to return to user
								(*Xvsalphalambda)[MAPXALL(i, a, n+NUMError+2)] =
										tolnew;
							}
						} else {                  // only done if realfolds>1
							memcpy(&((*Xvsalpha)[MAPXBEST(a, 0)]),
									&h2o4gpu_data.GetX()[0], n * sizeof(T));
							// Save error to return to user
							ErrorLOOP(ri)
								(*Xvsalpha)[MAPXBEST(a, n + ri)] =
										localerror[ri];
							// Save lambda to return to user
							(*Xvsalpha)[MAPXBEST(a, n+NUMError)] = lambda;
							// Save alpha to return to user
							(*Xvsalpha)[MAPXBEST(a, n+NUMError+1)] = alpha;
							// Save tolnew to return to user
							(*Xvsalpha)[MAPXBEST(a, n+NUMError+2)] = tolnew;
						}

						if (lambdatype == LAMBDATYPEPATH) {
							if (lambdastopearly>0) {
								if (scoring_history.size() >= 1) {
									double ratio = (norm
											- scoring_history.back()) / norm;

									double fracdof = 0.5; //USER parameter.
									//                    if((double)dof>fracdof*(double)(n)){ // only consider stopping if explored most degrees of freedom, because at dof~0-1 error can increase due to tolerance in solver.
									if (RELAXEARLYSTOP
											|| ratio > 0.0
													&& (double) dof
															> fracdof
																	* (double) (n)) { // only consider stopping if explored most degrees of freedom, because at dof~0-1 error can increase due to tolerance in solver.
															//                  fprintf(stderr,"ratio=%g dof=%zu fracdof*n=%g\n",ratio,dof,fracdof*n); fflush(stderr);
															// STOP EARLY CHECK
										int k = 3; //TODO: ask the user for this parameter
										double tolerance = 0.0; // stop when not improved over 3 successive lambdas (averaged over window 3) // NOTE: Don't use tolerance=0 because even for simple.txt test this stops way too early when error is quite high
										bool moreIsBetter = false;
										bool verbose =
												static_cast<bool>(VERBOSEENET); // true;
										if (stopEarly(scoring_history, k,
												tolerance, moreIsBetter,
												verbose, norm, &jump)) {
											break;
										}
									}
								}
							}

							//                fprintf(stderr,"doskiplambda=%d skiplambdaamount=%d\n",doskiplambda,skiplambdaamount);fflush(stderr);
							// if can skip over lambda, do so, but still print out the score as if constant for new lambda
							if (doskiplambda) {
								for (int ii = 0; ii < skiplambdaamount; ++ii) {
									i++;
									if (i >= nlambdalocal)
										break; // don't skip beyond existing lambda
									lambda = lambdaslocal[i];
									if (VERBOSEENET)
										Printmescore(fil);
#pragma omp critical
									{
										if (VERBOSEENET)
											Printmescore(stdout);
										if (VERBOSEANIM || verboseanimtriggered==1){
											Printmescoresimple(filerror);
											Printmescoresimple2(filvarimp);
										}
									}
								}
							}
						}

					} // over lambda(s)

					// store results
					int pickfi;
					if (lambdatype == LAMBDATYPEPATH)
						pickfi = fi; // variable lambda folds
					else
						pickfi = realfolds + fi; // fixed-lambda folds
					// store Error (thread-safe)
					alphaarray[pickfi][a] = tbestalpha;
					lambdaarray[pickfi][a] = tbestlambda;
					tolarray[pickfi][a] = tbesttol;
					ErrorLOOP(ri)
						errorarray[ri][pickfi][a] = tbesterror[ri];

					// if not doing folds, store best solution over all lambdas
					if (lambdatype == LAMBDATYPEPATH && nFolds < 2) {
						if (fi == 0) { // only store first fold for user
							memcpy(&((*Xvsalpha)[MAPXBEST(a, 0)]),
									&h2o4gpu_data.GetX()[0], n * sizeof(T)); // not quite best, last lambda TODO FIXME
							//                for(unsigned int iii=0; iii<n;iii++) fprintf(stderr,"Xvsalpha[%d]=%g\n",iii,(*Xvsalpha)[MAPXBEST(a,iii)]); fflush(stderr);
							// Save error to return to user
							ErrorLOOP(ri)
								(*Xvsalpha)[MAPXBEST(a, n + ri)] =
										tbesterror[ri];
							// Save lambda to return to user
							(*Xvsalpha)[MAPXBEST(a, n+NUMError)] = tbestlambda;
							// Save alpha to return to user
							(*Xvsalpha)[MAPXBEST(a, n+NUMError+1)] = tbestalpha;
							// Save tol to return to user
							(*Xvsalpha)[MAPXBEST(a, n+NUMError+2)] = tbesttol;
						}
					}

				}                // over folds
			}                // over alpha

#pragma omp barrier // barrier so alphaarray, lambdaarray, errorarray are filled and ready to be read by all threads
			int fistart;
			if (lambdatype == LAMBDATYPEPATH)
				fistart = 0; // variable lambda folds
			else
				fistart = realfolds; // fixed-lambda folds

			// get CV averaged Error and best solution (using shared memory arrays that are thread-safe)
			double bestalpha = 0;
			double bestlambda = 0;
			double besttol = std::numeric_limits<double>::max();
			double besterror[NUMError];
			ErrorLOOP(ri)
				besterror[ri] = std::numeric_limits<double>::max();
			for (size_t a = 0; a < nAlphas; ++a) { //alpha
				alphaarrayofa[a] = 0.0;
				lambdaarrayofa[a] = 0.0;
				tolarrayofa[a] = std::numeric_limits<double>::max();
				ErrorLOOP(ri)
					errorarrayofa[ri][a] = 0.0;
				for (size_t fi = fistart; fi < fistart + realfolds; ++fi) { //fold
					alphaarrayofa[a] += alphaarray[fi][a];
					lambdaarrayofa[a] += lambdaarray[fi][a];
#define MIN(a,b) ((a)<(b)?(a):(b))
					tolarrayofa[a] = MIN(tolarrayofa[a], tolarray[fi][a]); // choose common min tolerance
					ErrorLOOP(ri)
						errorarrayofa[ri][a] += errorarray[ri][fi][a];
				}
				// get average error over folds for this alpha
				alphaarrayofa[a] /= ((double) (realfolds));
				lambdaarrayofa[a] /= ((double) (realfolds));
				ErrorLOOP(ri)
					errorarrayofa[ri][a] /= ((double) (realfolds));
				if (errorarrayofa[owhicherror][a] < besterror[owhicherror]) {
					bestalpha = alphaarrayofa[a]; // get alpha for this case
					bestlambda = lambdaarrayofa[a]; // get lambda for this case
					besttol = tolarrayofa[a]; // get tol for this case
					ErrorLOOP(ri)
						besterror[ri] = errorarrayofa[ri][a]; // get best error as average for this alpha
				}
				if (VERBOSEENET) {
					if (lambdatype == LAMBDATYPEPATH && realfolds > 1)
						fprintf(stderr,
								"To use for last CV models: alpha=%g lambda=%g tol=%g\n",
								alphaarrayofa[a], lambdaarrayofa[a],
								tolarrayofa[a]);
					fflush(stderr);
				}
			}

			// print result (all threads have same result, so only need to print on one thread)
			if (me == 0 && VERBOSEENET)
				PrintmescoresimpleCV(stdout,lambdatype,bestalpha,bestlambda,besterror[0],besterror[1],besterror[2]);

		} // over lambdatype

		if (X0)
			delete[] X0;
		if (L0)
			delete[] L0;
		if (fil != NULL)
			fclose(fil);
	} // end parallel region

	///////////////////////
	//
	// report over all folds, cross-validated model, and over alphas
	//
	///////////////////////
	if (VERBOSEENET) {
		for (size_t fi = 0; fi < totalfolds; ++fi) { //fold
			for (size_t a = 0; a < nAlphas; ++a) { //alpha
				fprintf(stderr,
						"pass=%d fold=%zu alpha=%21.15g lambda=%21.15g errorTrain=%21.15g erroriValid=%21.15g errorValid=%21.15g\n",
						(fi >= realfolds ? 1 : 0),
						(fi >= realfolds ? fi - realfolds : fi),
						alphaarray[fi][a], lambdaarray[fi][a],
						errorarray[0][fi][a], errorarray[1][fi][a],
						errorarray[2][fi][a]);
				fflush(stderr);
			}
		}
	}

	// free any malloc's
	if (trainX && OLDPRED)
		free(trainX);
	if (trainY)
		free(trainY);
	if (validX && OLDPRED)
		free(validX);
	if (validY)
		free(validY);
	if (trainW)
		free(trainW);
	if (filerror != NULL) fclose(filerror);
	if (filvarimp != NULL) fclose(filvarimp);

	double tf = timer<double>();
	if (VERBOSEENET) {
		fprintf(stdout,
				"END SOLVE: type 1 mTrain %d n %d mValid %d twall %g tsolve(post-dataongpu) %g\n",
				(int) mTrain, (int) n, (int) mValid, tf - t, tf - t1me0);
		fflush(stdout);
	}
	if (flag) {
		fprintf(stderr, "Signal caught. Terminated early.\n");
		fflush(stderr);
		flag = 0; // set flag
	}
	return tf - t;
}

template<typename T>
double ElasticNetptr_predict(const char family, int sourceDev, int datatype, int sharedA,
		int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n,
		size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		T *alphas, T *lambdas,
		double tol, double tol_seek_factor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction, int max_iterations, int verbose,
		void *trainXptr, void *trainYptr, void *validXptr, void *validYptr,
		void *weightptr, int givefullpath, T **Xvsalphalambda, T **Xvsalpha,
		T **validPredsvsalphalambda, T **validPredsvsalpha, size_t *countfull,
		size_t *countshort, size_t *countmore) {


	// Adjust any parameters for user friendliness
	nAlphas = std::max(nAlphas,0); // At least zero alphas
	nLambdas = std::max(nLambdas,0); // At least zero Lambdas


	signal(SIGINT, my_function);
	signal(SIGTERM, my_function);
	int nlambda = nLambdas;

	FILE *filerror = NULL;
	if(VERBOSEANIM){
	// critical files for all threads
	char filename0[100];
	sprintf(filename0, "prederror.txt");
	filerror = fopen(filename0, "wt");
	if (filerror == NULL) {
		cerr << "Cannot open filename0=" << filename0 << endl;
		exit(0);
	}
	}
	if (VERBOSEENET) {
		cout << "Hardware: " << HARDWARE << endl;
	}

	// number of openmp threads = number of cuda devices to use
#ifdef _OPENMP
	int omt=omp_get_max_threads();
	//      omp_set_num_threads(MIN(omt,nGPUs));  // not necessary, but most useful mode so far
	omp_set_num_threads(nThreads);// not necessary, but most useful mode so far
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

	if (VERBOSEENET) {
		fprintf(stderr, "Before malloc validPreds\n");
		fflush(stderr);
	}

	// setup storage for returning results back to user
	if (givefullpath) {
		*validPredsvsalphalambda = (T*) calloc(
				*countfull / (n + NUMOTHER) * mValid, sizeof(T)); // +NUMOTHER for values of lambda, alpha, and tolerance
	} else { // only give back solution for optimal lambda after CV is done
		*validPredsvsalphalambda = NULL;
	}
	*validPredsvsalpha = (T*) calloc(*countshort / (n + NUMOTHER) * mValid,
			sizeof(T));
//	printf("inside Pred: countfull=%zu countshort=%zu\n",*countfull/(n+NUMOTHER)*mValid,*countshort/(n+NUMOTHER)*mValid); fflush(stdout);

	if (VERBOSEENET) {
		fprintf(stderr, "After malloc validPreds\n");
		fflush(stderr);
	}
	// for source, create class objects that creates cuda memory, cpu memory, etc.
	// This takes-in raw GPU pointer
	//  h2o4gpu::MatrixDense<T> Asource_(sourceDev, ord, mTrain, n, mValid, reinterpret_cast<T *>(trainXptr));
	// assume source thread is 0th thread (TODO: need to ensure?)
	int sourceme = sourceDev;
	if (VERBOSEENET) {
		fprintf(stderr, "Before Asource\n");
		fflush(stderr);
	}
	h2o4gpu::MatrixDense<T> Asource_(sharedA, sourceme, sourceDev, datatype,
			ord, mTrain, n, mValid, reinterpret_cast<T *>(trainXptr),
			reinterpret_cast<T *>(trainYptr), reinterpret_cast<T *>(validXptr),
			reinterpret_cast<T *>(validYptr), reinterpret_cast<T *>(weightptr));
	// now can always access A_(sourceDev) to get pointer from within other MatrixDense calls
	if (VERBOSEENET) {
		fprintf(stderr, "After Asource\n");
		fflush(stderr);
	}
	// Setup each thread's h2o4gpu
	double t = timer<double>();
	double t1me0;

	////////////////////////////////
	// PARALLEL REGION
#if USEPARALLEL != 0
#pragma omp parallel proc_bind(master)
#endif
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
		int me = 0;
#endif

		int blasnumber;
#ifdef HAVECUDA
		blasnumber=CUDA_MAJOR;
#else
		blasnumber = mklperthread; // roughly accurate for openblas as well
#endif

		// choose GPU device ID for each thread
		int wDev = gpu_id + (nGPUs > 0 ? me % nGPUs : 0);
		wDev = wDev % totalnGPUs;

		FILE *fil=NULL;
		if(VERBOSEANIM){
		// Setup file output
		char filename[100];
		sprintf(filename, "predme%d.%d.%s.%s.%d.%d.%d.txt", me, wDev, _GITHASH_,
				TEXTARCH, sharedA, nThreads, nGPUs);
		fil = fopen(filename, "wt");
		if (fil == NULL) {
			cerr << "Cannot open filename=" << filename << endl;
			exit(0);
		} else
			fflush(fil);
		}

		////////////
		//
		// create class objects that creates cuda memory, cpu memory, etc.
		//
		////////////
		double t0 = timer<double>();
		DEBUG_FPRINTF(fil,
				"Pred: Moving data to the GPU. Starting at %21.15g\n", t0);
#pragma omp barrier // not required barrier
		h2o4gpu::MatrixDense<T> A_(sharedA, me, wDev, Asource_);
#pragma omp barrier // required barrier for wDev=sourceDev so that Asource_._data (etc.) is not overwritten inside h2o4gpu_data(wDev=sourceDev) below before other cores copy data
		h2o4gpu::H2O4GPUDirect<T, h2o4gpu::MatrixDense<T> > h2o4gpu_data(
				sharedA, me, wDev, A_);
#pragma omp barrier // not required barrier
		double t1 = timer<double>();
		if (me == 0) { //only thread=0 times entire post-warmup procedure
			t1me0 = t1;
		}
		DEBUG_FPRINTF(fil,
				"Pred: Done moving data to the GPU. Stopping at %21.15g\n", t1);
		DEBUG_FPRINTF(fil, "Pred: Done moving data to the GPU. Took %g secs\n",
				t1 - t0);

		////////////////////////////////////////////////
		// BEGIN GLM
		DEBUG_FPRINTF(fil, "BEGIN SOLVE: %d\n", 0);

		int a, i;

		//////////////////////////////
		// LOOP OVER ALPHAS
		///////////////////////////////
#pragma omp for schedule(dynamic,1) collapse(1)
		for (a = 0; a < nAlphas; ++a) { //alpha search


			int nlambdalocal;
			if (givefullpath) { // then need to loop over same nlambda
				nlambdalocal = nlambda;
			} else {
				nlambdalocal = 1; // only 1 lambda
			}

			// LOOP over lambda
			for (i = 0; i < nlambdalocal; ++i) {

				// copy existing solution to X0
				T *X0 = new T[n]();
				if (givefullpath) {
					memcpy(X0, &((*Xvsalphalambda)[MAPXALL(i, a, 0)]),
							n * sizeof(T));
				} else {
					memcpy(X0, &((*Xvsalpha)[MAPXBEST(a, 0)]), n * sizeof(T));
				}

				// set X from X0
				h2o4gpu_data.SetInitX(X0);


				// compute predictions
				h2o4gpu_data.Predict();

				// Get valid prediction
				std::vector<T> validPreds(&h2o4gpu_data.GetvalidPreds()[0],
						&h2o4gpu_data.GetvalidPreds()[0] + mValid);

				//Compute inverse logit of predictions if family == 'logistic' to get actual probabilities.
				if(family == 'l'){
					std::transform(validPreds.begin(), validPreds.end(), validPreds.begin(),[](T i) -> T { return 1/(1+exp(-i)); });
				}

				T sdTrainY = 1.0; // TODO FIXME not impliemented yet
				T meanTrainY = 1.0; // TODO FIXME not impliemented yet
				// correct validPreds
				if (standardize) {
					for (size_t i = 0; i < mValid; ++i) { //row
						// reverse (fitted) standardization
						validPreds[i] *= sdTrainY; //scale
						validPreds[i] += meanTrainY; //intercept
					}
				}

				// save preds (exclusive set, unlike X)
				if (givefullpath) { // save all preds
					memcpy(&((*validPredsvsalphalambda)[MAPPREDALL(i, a, 0, mValid)]),
							&validPreds[0], mValid * sizeof(T));
				} else { // save only best pred per lambda
					memcpy(&((*validPredsvsalpha)[MAPPREDBEST(a, 0, mValid)]),
							&validPreds[0], mValid * sizeof(T));
				}

				// get validY so can compute Error
				//T *validY = NULL;
				//validY = (T *) malloc(sizeof(T) * mValid);
				T *validY = new T[mValid];
				int validYerror = Asource_.GetValidY(datatype, mValid, &validY);

				// Compute Error for predictions
				if (validYerror == 0) {

					T weightsvalid[mValid];
					for (size_t i = 0; i < mValid; ++i) { //row
						weightsvalid[i] = 1.0;
					}

					T validError = h2o4gpu::getError(weightsvalid, mValid,
							&validPreds[0], validY, family);
					if (standardize)
						validError *= sdTrainY;

					if (givefullpath) {
						// Save error to return to user
						int ri = VALIDError;
						(*Xvsalphalambda)[MAPXALL(i, a, n + ri)] = validError; // overwrite any old value done during fit
					} else {
						int ri = VALIDError;
						(*Xvsalpha)[MAPXBEST(a, n + ri)] = validError;
					}

					// report scores
					if (VERBOSEENET)
						Printmescore_predict(fil);
#pragma omp critical
					{
						if (VERBOSEENET)
							Printmescore_predict(stdout);
						if (VERBOSEANIM)
							Printmescoresimple_predict(filerror);
					}

				} else {
					// report scores
					if (VERBOSEENET)
						Printmescore_predictnovalid(fil);
#pragma omp critical
					{
						if (VERBOSEENET)
							Printmescore_predictnovalid(stdout);
						if (VERBOSEANIM)
							Printmescoresimple_predictnovalid(filerror);
					}

				}

				if (X0)
					delete[] X0;
				if (validY)
					delete[] validY;
			} // over lambda(s)

		} // over alpha

		if (fil != NULL)
			fclose(fil);
	} // end parallel region

	if (filerror != NULL) fclose(filerror);

	double tf = timer<double>();
	if (VERBOSEENET) {
		fprintf(stdout,
				"END PREDICT: type 1 mTrain %d n %d mValid %d twall %g tsolve(post-dataongpu) %g\n",
				(int) mTrain, (int) n, (int) mValid, tf - t, tf - t1me0);
		fflush(stdout);
	}
	if (flag) {
		fprintf(stderr, "Signal caught. Terminated early.\n");
		fflush(stderr);
		flag = 0; // set flag
	}
	return tf - t;
}

template double ElasticNetptr<double>(
		const char family, int dopredict, int sourceDev,
		int datatype, int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord,
		size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		double *alphas, double *lambdas,
		double tol,  double tolseekfactor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction, int max_iterations,
		int verbose, void *trainXptr, void *trainYptr, void *validXptr,
		void *validYptr, void *weightptr, int givefullpath,
		double **Xvsalphalambda, double **Xvsalpha,
		double **validPredsvsalphalambda, double **validPredsvsalpha,
		size_t *countfull, size_t *countshort, size_t *countmore);

template double ElasticNetptr<float>(const char family, int dopredict, int sourceDev, int datatype,
		int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain,
		size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		float *alphas, float *lambdas,
		double tol,  double tolseekfactor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction, int max_iterations,
		int verbose, void *trainXptr, void *trainYptr, void *validXptr,
		void *validYptr, void *weightptr, int givefullpath,
		float **Xvsalphalambda, float **Xvsalpha,
		float **validPredsvsalphalambda, float **validPredsvsalpha,
		size_t *countfull, size_t *countshort, size_t *countmore);

template double ElasticNetptr_fit<double>(const char family, int sourceDev, int datatype,
		int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain,
		size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		double *alphas, double *lambdas,
		double tol,  double tolseekfactor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction, int max_iterations,
		int verbose, void *trainXptr, void *trainYptr, void *validXptr,
		void *validYptr, void *weightptr, int givefullpath,
		double **Xvsalphalambda, double **Xvsalpha,
		double **validPredsvsalphalambda, double **validPredsvsalpha,
		size_t *countfull, size_t *countshort, size_t *countmore);

template double ElasticNetptr_fit<float>(const char family, int sourceDev, int datatype,
		int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain,
		size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		float *alphas, float *lambdas,
		double tol,  double tolseekfactor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction, int max_iterations,
		int verbose, void *trainXptr, void *trainYptr, void *validXptr,
		void *validYptr, void *weightptr, int givefullpath,
		float **Xvsalphalambda, float **Xvsalpha,
		float **validPredsvsalphalambda, float **validPredsvsalpha,
		size_t *countfull, size_t *countshort, size_t *countmore);

template double ElasticNetptr_predict<double>(const char family, int sourceDev, int datatype,
		int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain,
		size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		double *alphas, double *lambdas,
		double tol,  double tolseekfactor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction, int max_iterations,
		int verbose, void *trainXptr, void *trainYptr, void *validXptr,
		void *validYptr, void *weightptr, int givefullpath,
		double **Xvsalphalambda, double **Xvsalpha,
		double **validPredsvsalphalambda, double **validPredsvsalpha,
		size_t *countfull, size_t *countshort, size_t *countmore);

template double ElasticNetptr_predict<float>(const char family, int sourceDev, int datatype,
		int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain,
		size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		float *alphas, float *lambdas,
		double tol,  double tolseekfactor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction, int max_iterations,
		int verbose, void *trainXptr, void *trainYptr, void *validXptr,
		void *validYptr, void *weightptr, int givefullpath,
		float **Xvsalphalambda, float **Xvsalpha,
		float **validPredsvsalphalambda, float **validPredsvsalpha,
		size_t *countfull, size_t *countshort, size_t *countmore);

template<typename T>
int modelFree2(T *aptr) {
	free(aptr);
	return (0);
}

template int modelFree2<float>(float *aptr);
template int modelFree2<double>(double *aptr);

double elastic_net_ptr_double(const char family, int dopredict, int sourceDev, int datatype,
		int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain,
		size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		double *alphas, double *lambdas,
		double tol,  double tolseekfactor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction, int max_iterations,
		int verbose, void *trainXptr, void *trainYptr, void *validXptr,
		void *validYptr, void *weightptr, int givefullpath,
		double **Xvsalphalambda, double **Xvsalpha,
		double **validPredsvsalphalambda, double **validPredsvsalpha,
		size_t *countfull, size_t *countshort, size_t *countmore) {
	return ElasticNetptr<double>(family, dopredict, sourceDev, datatype, sharedA,
			nThreads, gpu_id, nGPUs, totalnGPUs, ord, mTrain, n, mValid, intercept, standardize,
			lambda_max, lambda_min_ratio, nLambdas, nFolds,
			nAlphas, alpha_min, alpha_max,
			alphas, lambdas,
			tol, tolseekfactor,
			lambdastopearly, glmstopearly, stopearlyerrorfraction, max_iterations, verbose, trainXptr,
			trainYptr, validXptr, validYptr, weightptr, givefullpath,
			Xvsalphalambda, Xvsalpha, validPredsvsalphalambda,
			validPredsvsalpha, countfull, countshort, countmore);
}
double elastic_net_ptr_float(const char family, int dopredict, int sourceDev, int datatype,
		int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain,
		size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max,
		float *alphas, float *lambdas,
		double tol,  double tolseekfactor,
		int lambdastopearly, int glmstopearly, double stopearlyerrorfraction, int max_iterations,
		int verbose, void *trainXptr, void *trainYptr, void *validXptr,
		void *validYptr, void *weightptr, int givefullpath,
		float **Xvsalphalambda, float **Xvsalpha,
		float **validPredsvsalphalambda, float **validPredsvsalpha,
		size_t *countfull, size_t *countshort, size_t *countmore) {
	return ElasticNetptr<float>(family, dopredict, sourceDev, datatype, sharedA,
			nThreads, gpu_id, nGPUs, totalnGPUs, ord, mTrain, n, mValid, intercept, standardize,
			lambda_max, lambda_min_ratio, nLambdas, nFolds,
			nAlphas, alpha_min, alpha_max,
			alphas, lambdas,
			tol, tolseekfactor,
			lambdastopearly, glmstopearly, stopearlyerrorfraction, max_iterations, verbose, trainXptr,
			trainYptr, validXptr, validYptr, weightptr, givefullpath,
			Xvsalphalambda, Xvsalpha, validPredsvsalphalambda,
			validPredsvsalpha, countfull, countshort, countmore);
}

int modelfree2_float(float *aptr) {
	return modelFree2<float>(aptr);
}
int modelfree2_double(double *aptr) {
	return modelFree2<double>(aptr);
}

}
