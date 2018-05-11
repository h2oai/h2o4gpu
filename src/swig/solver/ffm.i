/* File : ffm.i */
%{
#include "../../include/data/ffm/data.h"
#include "../../include/solver/ffm_api.h"
%}

%include carrays.i

%rename("params_ffm") ffm::Params;

%apply (float *INPLACE_ARRAY1) {float *predictions, float *w};
%apply (double *INPLACE_ARRAY1) {double *predictions, double *w};

%include "../../include/data/ffm/data.h"
%include "../../include/solver/ffm_api.h"

%template(doubleNode) ffm::Node<double>;
%template(floatNode) ffm::Node<float>;

%template(doubleRow) ffm::Row<double>;
%template(floatRow) ffm::Row<float>;

%template(doubleDataset) ffm::Dataset<double>;
%template(floatDataset) ffm::Dataset<float>;

%array_class(ffm::Row<float>, RowFloatArray)
%array_class(ffm::Row<double>, RowDoubleArray)

%array_class(ffm::Node<float>, NodeFloatArray)
%array_class(ffm::Node<double>, NodeDoubleArray)