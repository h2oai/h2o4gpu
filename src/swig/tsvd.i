/* File : tsvd.i */
%{
#include "../gpu/tsvd/tsvd.h"
%}

%rename("params_tsvd") tsvd::params;

%include "../gpu/tsvd/tsvd.h"
