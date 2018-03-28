/* File : pca.i */
%{
#include "../gpu/pca/pca.h"
%}

%rename("params_pca") pca::params;

%include "../gpu/pca/pca.h"