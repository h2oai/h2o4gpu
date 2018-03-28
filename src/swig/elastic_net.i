/* File : elastic_net.i */
%{
extern int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                             const double* trainX, const double* trainY, const double* validX, const double* validY, const double *weight,
                             void**a, void**b, void**c, void**d, void **e);

extern int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                            const float* trainX, const float* trainY, const float* validX, const float* validY, const float *weight,
                            void**a, void**b, void**c, void**d, void **e);
%}

extern int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                             const double* trainX, const double* trainY, const double* validX, const double* validY, const double *weight,
                             void**a, void**b, void**c, void**d, void **e);

extern int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                            const float* trainX, const float* trainY, const float* validX, const float* validY, const float *weight,
                            void**a, void**b, void**c, void**d, void **e);