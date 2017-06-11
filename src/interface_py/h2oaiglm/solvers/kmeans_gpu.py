from h2oaiglm.libs.kmeans_gpu import h2oaiKMeansGPU

if not h2oaiKMeansGPU:
    print('\nWarning: Cannot create a H2OAIKMeans GPU Solver instance without linking Python module to a compiled H2OAIGLM GPU library')
    print('> Setting h2oaiglm.KMeansGPU=None')
    print('> Add CUDA libraries to $PATH and re-run setup.py\n\n')
    KMeansGPU=None
else:
    class KMeansGPUSolver(object):
        def __init__(self, lib, nGPUs, ordin, k):
            assert lib and (lib==h2oaiglmElasticNetGPU)
        self.lib=lib
        self.nGPUs=nGPUs
        self.sourceDev=0 # assume Dev=0 is source of data for upload_data
        self.ord=ord(ordin)
        self.k=k

    # def upload_data(self, sourceDev, trainX):
    #     mTrain = trainX.shape[0]
    #     n = trainX.shape[1]
    #     a = c_void_p(0)
    #     if (trainX.dtype==np.float64):
    #         print("Detected np.float64");sys.stdout.flush()
    #         self.double_precision=1
    #         A = cptr(trainX,dtype=c_double)
    #         status = self.lib.make_ptr_double_kmeans(c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_int(self.ord), A, pointer(a))
    #
    #     elif (trainX.dtype==np.float32):
    #         print("Detected np.float32");sys.stdout.flush()
    #         self.double_precision=0
    #         A = cptr(trainX,dtype=c_float)
    #         status = self.lib.make_ptr_float_kmeans(c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_int(self.ord), A, pointer(a))
    #     else:
    #         print("Unknown numpy type detected")
    #         print(trainX.dtype)
    #         sys.stdout.flush()
    #         exit(1)
    #
    #     assert status==0, "Failure uploading the data"
    #     print(a)
    #     return a
    #
    # # sourceDev here because generally want to take in any pointer, not just from our test code
    # def fit(self, sourceDev, mTrain, n, precision, a):
    #     # not calling with self.sourceDev because want option to never use default but instead input pointers from foreign code's pointers
    #     if hasattr(self, 'double_precision'):
    #         whichprecision=self.double_precision
    #     else:
    #         whichprecision=precision
    #     #
    #     if (whichprecision==1):
    #         print("double precision fit")
    #         self.lib.kmeans_ptr_double(
    #             c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
    #             c_size_t(mTrain), c_size_t(n), c_size_t(mValid),c_int(self.intercept), c_int(self.standardize),
    #             c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
    #             a, b, c, d, e)
    #     else:
    #         print("single precision fit")
    #         self.lib.kmeans_ptr_float(
    #             c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
    #             c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.intercept), c_int(self.standardize),
    #             c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
    #             a, b, c, d, e)
    #     print("Done with fit")



    class KMeansGPU(object):
        def __init__(self, nGPUs, ord, k):
            self.solver = KMeansGPUSolver(h2oaiKMeansGPU, nGPUs, ord, k)

        # def upload_data_ptr(self, sourceDev, trainX):
        # 	return self.solver.upload_data(sourceDev, trainX)
        #
        # def fit_ptr(self, sourceDev, mTrain, n, a):
        # 	return self.solver.fit(sourceDev, mTrain, n, a)

        def fit(self, mTrain, n, data, k):
            res = c_void_p(0)
            self.lib.make_ptr_float_kmeans(self.solver.nGPUs, mTrain, n, self.solver.ord, data, k, pointer(res))
            self.centroids = np.iter(cast(res, float_p), n, dtype=np.float32)
