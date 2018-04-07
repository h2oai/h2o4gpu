# How to implement a new solver.

Here's a step-by-step tutorial on how to add new functionality starting from scratch (we'll be adding a new solver, in case you're adding a new data structure, metrics, utils etc. just substitute "solver" in names and path appropriately using common sense).

### API Header

Add a header file in `src/include` or an approriate subfolder. This header file should expose *only* whatever you want the user facing API to expose (methods, structures etc.). Nothing more. For example for a new solver add:

src/include/solver/solver_name_api.h

```cpp
#pragma once

typedef struct solver_params {
  int n;
  int m;
  int k;
} solver_params;

/**
 * Documentation goes here
 *
 * \param[in]		data
 * \param[out]		labels
 * \param[in,out]	centroids
 */
void solver_name_fit_float(const float *data, float *labels, float *centroids, solver_params _params);
void solver_name_fit_double(const double *data, double *labels, double *centroids, solver_params _params);

/**
 * Documentation goes here
 *
 * \param[in]		data
 * \param[in]		centroids
 * \param[out]		labels
 */
void solver_name_predict_float(const float *data, const float *centroids, float *labels, solver_params _params);
void solver_name_predict_double(const double *data, const double *centroids, double *labels, solver_params _params);
```

### Solver implementation

Add the common C/CPP code in `src/common`. Try to DRY as much code as possible and only extract CPU/GPU specific code into separate files (as shown in following sections), e.g.:

src/common/solver/solver_name.cpp

```
#include "../include/solver/solver_name.h"
#include "../include/solver/solver_name_api.h"
 
void initializeCentroids(Matrix<T> &X, Matrix<T> &centroids) {
	// copy k rows from X to centroids
}
 
void calculateDistance(Matrix<T> &distances, Matrix<T> &X, Matrix<T> &centroids) {
	// calculate the distance between each row in X and each row in centroids, put result in distances
}
 
template<typename T>
void solver_name_fit(const T *data, T *labels, T *centroids, solver_params _params) {
    Matrix<T> X(data, _params.n, _params.m);
    Matrix<T> centroids(centroids, _params.k, _params.m);
    Matrix<int> labels_matrix(_params.n, 1);
 
    initializeCentroids(X, centroids);
    Matrix<T> distances(_params.n, _params.k);
    for(int i = 0; i < _params.iterations && totalMoved > _params.threshold; i++) {
        calculateDistance(distances, X, centroids);
        double totalMoved = recalculateCentroids(centroids, distances);
        relabel(labels_matrix, distances);
    }
    labels_matrix.copy_to(labels);
}
 
template<typename T>
void solver_name_predict(const T *data, T *labels, T *centroids, solver_params _params) {
    Matrix<T> X(data, _params.n, _params.m);
    Matrix<T> centroids(centroids, _params.k, _params.m);
    Matrix<int> labels_matrix(_params.n, 1);
    Matrix<T> distances(_params.n, _params.k);
 
    calculateDistance(distances, X, centroids);
    relabel(labels_matrix, distances);
 
    labels_matrix.copy(labels);
}
 
template
void solver_name_fit<float>(const float *data, float *labels, float *centroids, solver_params _params);
 
template
void solver_name_fit<double>(const double *data, double *labels, double *centroids, solver_params _params);
 
template
void solver_name_predict<float>(const float *data, float *labels, float *centroids, solver_params _params);
 
template
void solver_name_predict<double>(const double *data, double *labels, double *centroids, solver_params _params);
 
void solver_name_fit_float(const float *data, float *labels, float *centroids, solver_params _params) {
	solver_name_fit(data, labels, centroids, _params);
}
 
void solver_name_fit_double(const double *data, double *labels, double *centroids, solver_params _params) {
	solver_name_fit(data, labels, centroids, _params);
}
 
void solver_name_predict_float(const float *data, const float *centroids, float *labels, solver_params _params) {
	solver_name_predict(data, labels, centroids, _params);
}
 
void solver_name_predict_double(const double *data, const double *centroids, double *labels, solver_params _params) {
	solver_name_predict(data, labels, centroids, _params);
}
 
```

### Solver related header files

Add headers for files which will require separate CPU/GPU implementation like for example:

##### Data structures:

Since we will need a data structure for CPU and GPU.

src/include/matrix/matrix.h

```
#pragma once

* \class	Matrix
*
* \brief	Matrix type. Doc goes here.
*
*/
template <typename T>
class Matrix {
	size_t _n;
	size_t _m;

	T* _data;

public:
	Matrix(size_t n, size_t m);
	Matrix(T* data, size_t n, size_t m);
	
	void copy_to(T* dst);
}
```

##### Solver specific methods/classes:

Since parts of the code will require cuBLAS/thrust calls on GPU and parts BLAS/std on CPU etc.

src/include/solver/solver_name.h

```
#pragma once

#include "matrix/matrix.h"

/**
 * Documentation goes here
 *
 * \param[in,out]		labels
 * \param[in]			distances
 */
void relabel(Matrix<T> &labels, const Matrix<T> &distances);
```

### CPU/GPU code separation

Add CPU/GPU specific implementations of the above headers. 

##### CPU

CPU code goes in `src/cpu/*`.

* Sample Matrix class CPP impl used in our solver:

src/cpu/matrix/matrix.cpp
```
#include "../include/matrix/matrix.h

template <typename T>
Matrix<T>::Matrix(T* data, size_t n, size_t m) {
	_n = n;
	_m = m;
	_data = data;
}

template <typename T>
void Matrix<T>::copy_to(T* dst) {
	// CPU specific impl goes here
}

template class Matrix<float>;
template class Matrix<double>;
```

* Sample solver specific method for CPU:

src/cpu/solver/solver_name.cpp

```
#include "../include/solver/solver_name.h"

void relabel(Matrix<T> &labels, const Matrix<T> &distances) {
	// iterate using standard CPP methods over distances, find min index for each row
	// set it as new label in the corresponding position in labels 
}
```

##### GPU

GPU code goes in `src/gpu/*`

* Sample Matrix class CUDA impl used in our solver:

src/gpu/matrix/matrix.cu
```
#include "../include/matrix/matrix.h

template <typename T>
Matrix<T>::Matrix(T* data, size_t n, size_t m) {
	_n = n;
	_m = m;
	
	safe_cuda(cudaMalloc(&_data, _n * _m * sizeof(T)));
	
	thrust::copy(data, data + n * m, thrust::device_pointer_cast(_data));
}

template <typename T>
void Matrix<T>::copy_to(T* dst) {
	// GPU specific impl goes here
}

template class Matrix<float>;
template class Matrix<double>;
```

* Sample solver specific method for GPU:

src/gpu/solver/solver_name.cpp

```
#include "../include/solver/solver_name.h"

void relabel(Matrix<T> &labels, const Matrix<T> &distances) {
	// iterate using CUDA methods (for example Thrust) over distances, find min index for each row
	// set it as new label in the corresponding position in labels 
}
```

### SWIG

Add SWIG interface file:

src/swig/solver/solver_name.i

```
/* File : solver_name.i */
%{
#include "../../include/solver/solver_name.h"
%}

/* In case of 1D arrays use _ARRAY1 variants */
/* For other mappings consult out other interface files, swig doc or numpy.i doc*/
%apply (float *IN_ARRAY2) {float *data};
%apply (float *OUT_ARRAY2) {float *labels};
%apply (float *INPLACE_ARRAY2) {float *centroids};

%apply (double *IN_ARRAY2) {double *data};
%apply (double *OUT_ARRAY2) {double *labels};
%apply (double *INPLACE_ARRAY2) {double *centroids};

%include "../../include/solver/solver_name.h"
```

### Add the interface file

Include the interface file in (either both or only one):

* src/swig/ch2o4gpu_cpu.i - if you provide CPU implementation
* src/swig/ch2o4gpu_gpu.i - if you provide GPU implementation

```
%include "solver/solver_name.i"
```

### Python

Add Python wrapper files.

src/interface_py/h2o4gpu/libs/solvers/solver_name.py
```
# If your solver is not implemented by ScikitLearn or implements it 100%, drop the H2O from the name
class solver_nameH2O(object):
    """Doc goes here.

    Parameters
    ----------
    param_name: type, default=default_val
        Doc goes here
    """
    
    def __init__(self, k, n_gpus = -1):
        self.param_name = param_name
        from ..libs.lib_utils import get_lib
        from ..util.gpu import device_count
        n_gpus, devices = device_count(n_gpus)
        self.lib = get_lib(n_gpus, devices) # Grabs the SWIG generated wrapper which exposes C methods
        
        self.k = k
        
        self.centroids = None
        self.labels = None
 
    def fit(self, X, y=None):
        """Fit

        :param: X {array-like}, shape (n_samples, n_features)
        	Doc goes here.

        :param y
        	Ignored

        :returns self : object
        """
        params = lib.solver_params()
        params.n = X.shape[0]
        params.m = X.shape[1]
        params.k = self.k
        
        c_method = lib.solver_name_fit_float if X.dtype == np.float32 else lib.solver_name_fit_double
        
        self.centroids = np.empty((params.k, params.m), dtype=X.dtype)
        self.labels = np.empty(params.n, dtype=np.int32)
        
        c_method(X, self.labels, self.centroids, params) 
        
        return self
        
    def predict(self, X):
        # TODO implement
 
# If you are implementing a solver which is implemented by ScikitLearn and your implementation doesn't handle certain parameter SKLearn does, write a wrapper which detects it and chooses your impl or SKLearn
class solver_name(object):

```

### Java

Add Java wrapper files - *coming soon*.

### Tests

Add tests! Currently adding Python tests in `tests/python/open_data` is the easiest way. C/C++/CUDA tests coming soon.