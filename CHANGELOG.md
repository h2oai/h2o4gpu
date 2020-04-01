## Release 0.4.1
* xgboost fix output_margin parameter [#834](https://github.com/h2oai/h2o4gpu/pull/834)
* Downgrade LightGBM to last stable version(2.2.x)

## Release 0.4.0
* ARMA/ARIMA [#797](https://github.com/h2oai/h2o4gpu/pull/797)
* Random state seed for matrix factorization [#803](https://github.com/h2oai/h2o4gpu/pull/803)
* Update xgboost while keeping an old version(0.9) to allow convert old pickled model [#822](https://github.com/h2oai/h2o4gpu/pull/822)
* Update lightgbm
* Remove a lot dependencies and make them less strict [#826](https://github.com/h2oai/h2o4gpu/pull/826) [#800](https://github.com/h2oai/h2o4gpu/pull/800/files)
* CUDA-10.1 support
* Various bug fixes and dependencies update

## Release 0.3.2
* Fix K-means demo for multi-GPU [#630](https://github.com/h2oai/h2o4gpu/issues/630)
* CUDA-10
* Use gcc to build LightGBM on ppc64le

## Release 0.3.1
* Fix memory leak [#175](https://github.com/h2oai/h2o4gpu/issues/175) [#204](https://github.com/h2oai/h2o4gpu/issues/204)
* Static linking with CUDA dependencies
* CUDA forward compatibility
* Update NCCL to 2.4
* Update arrow to 0.12.0
* Update XGBoost
* Update numpy, scipy and scikit-learn
* Add experimental Matrix Factorization [#729](https://github.com/h2oai/h2o4gpu/pull/729)



## Release 0.3.0
* Conda build added
* Power 8 and Power 9 builds
* cuda8, cuda9, cuda92 builds
* Include LightGBM in cpu and gpu modes
* Update XGBoost
* Updates for pygdf/arrow and update associated demo notebooks
* Various bug fixes


## Release 0.2.0
* R API
* Intel DAAL initial support
* tSVD improvements

## Release 0.1.0
New algorithms:

* Truncated SVD
* PCA

Improvements to existing algorithms:
* KMeans|| for KMeans
* Performance improvements for KMeans
* Performance improvements for XGBoost