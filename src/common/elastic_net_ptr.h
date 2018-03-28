/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once
#include <stddef.h>
#include <stdio.h>
#include <limits>
#include <vector>
#include <cassert>
#include <iostream>
#include <random>

#include "matrix/matrix_dense.h"
#include "h2o4gpuglm.h"
#include "timer.h"
#include <omp.h>
#include <cmath>

namespace h2o4gpu {

/**
 * Calculate error based on family.
 *
 * If family == 'logistic',then compute logloss else compute rmse
 *
 * @param len Length of outcome vector
 * @param predicted Predictions made by solver
 * @param actual Actual `Y` values
 * @param family Family sent to solver. Default is `elasticnet`. If not, then assume `logistic` and compute `logloss`
 */
template<typename T>
T getError(size_t len, const T *predicted, const T *actual, const char family) {
	if (family == 'e') {
		double rmse = 0;
		for (size_t i = 0; i < len; ++i) {
			double d = predicted[i] - actual[i];
			rmse += d * d;
		}
		rmse /= (double) len;
		return static_cast<T>(std::sqrt(rmse));
	} else { //logistic
		double logloss = 0;
		for (size_t i = 0; i < len; ++i) {
			double d = 0;
			if (predicted[i] != actual[i]) {
				double x = std::min(
						std::max(1e-15, static_cast<double>(predicted[i])),
						1 - 1e-15);
				d = -1 * (actual[i] * log(x) + (1 - actual[i]) * log(1 - x));
			}
			logloss += d;
		}
		logloss /= (double) len;
		return static_cast<T>(logloss);
	}
}

/**
 * Calculate error based on family.
 *
 * If family == 'logistic',then compute logloss else compute rmse
 *
 * @param weights Weight vector given to observations
 * @param len Length of outcome vector
 * @param predicted Predictions made by solver
 * @param actual Actual `Y` values
 * @param family Family sent to solver. Default is `elasticnet`. If not, then assume `logistic` and compute `logloss`
 */
template<typename T>
T getError(const T*weights, size_t len, const T *predicted, const T *actual,
		const char family) {
	if (family == 'e') {
		double weightsum = 0;
		for (size_t i = 0; i < len; ++i) {
			weightsum += weights[i];
		}

		double rmse = 0;
		for (size_t i = 0; i < len; ++i) {
			double d = predicted[i] - actual[i];
			rmse += d * d * weights[i];
		}

		rmse /= weightsum;
		return static_cast<T>(std::sqrt(rmse));
	} else { //logistic
		double weightsum = 0;
		for (size_t i = 0; i < len; ++i) {
			weightsum += weights[i];
		}
		double logloss = 0;
		for (size_t i = 0; i < len; ++i) {
			double d = 0;
			if (predicted[i] != actual[i]) {
				double x = std::min(
						std::max(1e-15, static_cast<double>(predicted[i])),
						1 - 1e-15);
				d = -1 * (actual[i] * log(x) + (1 - actual[i]) * log(1 - x))
						* weights[i];
			}
			logloss += d;
		}
		logloss /= weightsum;
		return static_cast<T>(logloss);
	}

}

/**
 * Calculate error based on family.
 *
 * If family == 'logistic',then compute logloss else compute rmse
 *
 * @param offset Offset vector given to observations
 * @param weights Weight vector given to observations
 * @param len Length of outcome vector
 * @param predicted Predictions made by solver
 * @param actual Actual `Y` values
 * @param family Family sent to solver. Default is `elasticnet`. If not, then assume `logistic` and compute `logloss`
 */
template<typename T>
T getError(const T offset, const T*weights, size_t len, const T *predicted,
		const T *actual, const char family) {
	if (family == 'e') {
		double weightsum = 0;
		for (size_t i = 0; i < len; ++i) {
			weightsum += offset - weights[i];
		}

		double rmse = 0;
		for (size_t i = 0; i < len; ++i) {
			double d = predicted[i] - actual[i];
			rmse += d * d * (offset - weights[i]);
		}

		rmse /= weightsum;
		return static_cast<T>(std::sqrt(rmse));
	} else { //logistic
		double weightsum = 0;
		for (size_t i = 0; i < len; ++i) {
			weightsum += offset - weights[i];
		}
		double logloss = 0;
		for (size_t i = 0; i < len; ++i) {
			double d = 0;
			if (predicted[i] != actual[i]) {
				double x = std::min(
						std::max(1e-15, static_cast<double>(predicted[i])),
						1 - 1e-15);
				d = -1 * (actual[i] * log(x) + (1 - actual[i]) * log(1 - x))
						* (offset - weights[i]);
			}
			logloss += d;
		}
		logloss /= weightsum;
		return static_cast<T>(logloss);
	}

}

// C++ program for implementation of Heap Sort
#define mysize_t int
template<typename T>
void swap(T arr[], mysize_t arrid[], mysize_t a, mysize_t b) {
	T t = arr[a];
	arr[a] = arr[b];
	arr[b] = t;
	mysize_t tid = arrid[a];
	arrid[a] = arrid[b];
	arrid[b] = tid;
}
template<typename T>
void heapify(int whichmax, mysize_t a, T arr[], mysize_t arrid[], mysize_t n) {
	mysize_t left = 2 * a + 1;
	mysize_t right = 2 * a + 2;
	mysize_t min = a;
	if (whichmax == 0) { // max
		if (left < n) {
			if (arr[min] > arr[left])
				min = left;
		}
		if (right < n) {
			if (arr[min] > arr[right])
				min = right;
		}
	} else { // max abs
		if (left < n) {
			if (std::abs(arr[min]) > std::abs(arr[left]))
				min = left;
		}
		if (right < n) {
			if (std::abs(arr[min]) > std::abs(arr[right]))
				min = right;
		}
	}
	if (min != a) {
		swap(arr, arrid, a, min);
		heapify(whichmax, min, arr, arrid, n);
	}
}
template<typename T>
void heapSort(int whichmax, T arr[], mysize_t arrid[], mysize_t n) {
	mysize_t a; //cout<<"okk";
	for (a = (n - 2) / 2; a >= 0; a--) {
		heapify(whichmax, a, arr, arrid, n);
	}

	for (a = n - 1; a >= 0; a--) {
		swap(arr, arrid, 0, a);
		heapify(whichmax, 0, arr, arrid, a);
	}
}
template<typename T>
void printArray(T arr[], mysize_t arrid[], mysize_t k) {
	std::cout << "LARGEST: ";
	for (mysize_t i = 0; i < k; i++) {
		fprintf(stdout, "%d %21.15g ", arrid[i], arr[i]);
		fflush(stdout);
	}
	std::cout << "\n";

}

template<typename T>
void topk(int whichmax, T arr[], mysize_t arrid[], mysize_t n, mysize_t k,
mysize_t *whichbeta, T *valuebeta) {
	T arr1[k];
	mysize_t arrid1[k];
	mysize_t a;
	for (a = 0; a < k; a++) {
		arr1[a] = arr[a];
		arrid1[a] = arrid[a];
	}
	for (a = (k - 2) / 2; a >= 0; a--) {
		heapify(whichmax, a, arr1, arrid1, k);
	}
	for (a = k; a < n; a++) {
		if (arr1[0] < arr[a]) {
			arr1[0] = arr[a];
			arrid1[0] = arrid[a];
			heapify(whichmax, 0, arr1, arrid1, k);
		}
	}
	heapSort(whichmax, arr1, arrid1, k);
#ifdef DEBUG
	printArray(arr1,arrid1,k); // DEBUG
#endif

	for (mysize_t i = 0; i < k; i++) {
		whichbeta[i] = arrid1[i];
		valuebeta[i] = arr1[i];
	}

}
/* A utility function to print array of size n */

// Driver program
template<typename T>
int topkwrap(int whichmax, mysize_t n, mysize_t k, T arr[], mysize_t *whichbeta,
		T *valuebeta) {
	mysize_t arrid[n];
	for (int i = 0; i < n; i++)
		arrid[i] = i;
	//cout<<"okk";
	topk(whichmax, arr, arrid, n, k, whichbeta, valuebeta);
	//   heapSort(whichmax, arr, arrid,n);

	//   cout << "Sorted array is \n";
	//   printArray(arr, arrid,n);

	return (0);
}

template int topkwrap<double>(int whichmax, mysize_t n, mysize_t k,
		double arr[], mysize_t *whichbeta, double *valuebeta);
template int topkwrap<float>(int whichmax, mysize_t n, mysize_t k, float arr[],
mysize_t *whichbeta, float *valuebeta);

// Elastic Net
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda \alpha ||x||_1 + \lambda 1-\alpha ||x||_2
//
// for many values of \lambda and multiple values of \alpha
// See <h2o4gpu>/matlab/examples/lasso_path.m for detailed description.
// m and n are training data size

template<typename T>
double ElasticNetptr(const char family, int dopredict, int sourceDev,
		int datatype, int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord,
		size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max, T *alphas, T *lambdas,
		double tol, double tolseekfactor, int lambdastopearly, int glmstopearly,
		double glmstopearlyerrorfraction, int max_iterations, int verbose,
		void *trainXptr, void *trainYptr, void *validXptr, void *validYptr,
		void *weightptr, int givefullpath, T **Xvsalphalambda, T **Xvsalpha,
		T **validPredsvsalphalambda, T **validPredsvsalpha, size_t *countfull,
		size_t *countshort, size_t *countmore);
template<typename T>
double ElasticNetptr_fit(const char family, int sourceDev, int datatype,
		int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain,
		size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max, T *alphas, T *lambdas,
		double tol, double tolseekfactor, int lambdastopearly, int glmstopearly,
		double glmstopearlyerrorfraction, int max_iterations, int verbose,
		void *trainXptr, void *trainYptr, void *validXptr, void *validYptr,
		void *weightptr, int givefullpath, T **Xvsalphalambda, T **Xvsalpha,
		T **validPredsvsalphalambda, T **validPredsvsalpha, size_t *countfull,
		size_t *countshort, size_t *countmore);
template<typename T>
double ElasticNetptr_predict(const char family, int sourceDev, int datatype,
		int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain,
		size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max, T *alphas, T *lambdas,
		double tol, double tolseekfactor, int lambdastopearly, int glmstopearly,
		double glmstopearlyerrorfraction, int max_iterations, int verbose,
		void *trainXptr, void *trainYptr, void *validXptr, void *validYptr,
		void *weightptr, int givefullpath, T **Xvsalphalambda, T **Xvsalpha,
		T **validPredsvsalphalambda, T **validPredsvsalpha, size_t *countfull,
		size_t *countshort, size_t *countmore);

template<typename T>
int makePtr_dense(int sharedA, int me, int wDev, size_t m, size_t n,
		size_t mValid, const char ord, const T *data, const T *datay,
		const T *vdata, const T *vdatay, const T *weight, void **_data,
		void **_datay, void **_vdata, void **_vdatay, void **_weight);

template<typename T>
int modelFree2(T *aptr);

double elastic_net_ptr_double(const char family, int dopredict, int sourceDev,
		int datatype, int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord,
		size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max, double *alphas,
		double *lambdas, double tol, double tolseekfactor, int lambdastopearly, int glmstopearly,
		double glmstopearlyerrorfraction, int max_iterations, int verbose,
		void *trainXptr, void *trainYptr, void *validXptr, void *validYptr,
		void *weightptr, int givefullpath, double **Xvsalphalambda,
		double **Xvsalpha, double **validPredsvsalphalambda,
		double **validPredsvsalpha, size_t *countfull, size_t *countshort,
		size_t *countmore);
double elastic_net_ptr_float(const char family, int dopredict, int sourceDev,
		int datatype, int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord,
		size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max, float *alphas,
		float *lambdas, double tol, double tolseekfactor, int lambdastopearly, int glmstopearly,
		double glmstopearlyerrorfraction, int max_iterations, int verbose,
		void *trainXptr, void *trainYptr, void *validXptr, void *validYptr,
		void *weightptr, int givefullpath, float **Xvsalphalambda,
		float **Xvsalpha, float **validPredsvsalphalambda,
		float **validPredsvsalpha, size_t *countfull, size_t *countshort,
		size_t *countmore);

int modelfree_double(double *aptr);
int modelfree_float(float *aptr);

}
