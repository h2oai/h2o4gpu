/* File : kmeans.i */
%{
#include "../../include/solver/kmeans.h"
%}

%apply (float *IN_ARRAY1) {float *srcdata, float *centroids};
%apply (float **INPLACE_ARRAY1) {float **pred_centroids, float **preds};

%apply (double *IN_ARRAY1) {double *srcdata, double *centroids};
%apply (double **INPLACE_ARRAY1) {double **pred_centroids, double **preds};

%apply (int **INPLACE_ARRAY1) {int **pred_labels};

%include "../../include/solver/kmeans.h"