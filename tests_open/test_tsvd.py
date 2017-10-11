import numpy as np
import time
from sklearn.decomposition import TruncatedSVD as sklearnsvd
from h2o4gpu.solvers import TruncatedSVD
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip

#Randomized scikit impl
#svd = TruncatedSVD(algorithm = "randomized", n_components=99, random_state=42, n_iter=5, tol=0.0)
X = np.array([[1, 2, 3], [4, 5, 6], [7,8,9], [10,11,12]], np.float32)
#X = np.random.rand(5000000,10)
k = 2

#Exact scikit impl
svd = sklearnsvd(algorithm = "arpack", n_components=k, random_state=42)

print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
print("Original X Matrix")
print(X)
print("\n")
print("tsvd run")
start_time = time.time()
trunc = TruncatedSVD(n_components=k) #Not really using k yet...
trunc.fit(X)
end_time = time.time() - start_time
print("Total time for tsvd is " + str(end_time))
print("tsvd Singular Values")
print(trunc.singular_values_)
print("tsvd Components (V^T)")
print(trunc.components_)
print("tsvd Explained Variance")
print(trunc.explained_variance_)
print("tsvd Explained Variance Ratio")
print(trunc.explained_variance_ratio_)

print("\n")
print("sklearn run")
start_sk = time.time()
svd.fit(X)
end_sk = time.time() - start_sk
print("Total time for sklearn is " + str(end_sk))
print("Sklearn Singular Values")
print(svd.singular_values_)
print("Sklearn Components (V^T)")
print(svd.components_)
print("Sklearn Explained Variance")
print(svd.explained_variance_)
print("Sklearn Explained Variance Ratio")
print(svd.explained_variance_ratio_)
#
# print("\n")
# print("tsvd U matrix")
# print(trunc.U)
# print("tsvd V^T")
# print(trunc.components_)
# print("tsvd Sigma")
# print(trunc.singular_values_)
# print("tsvd U * Sigma")
# x_tsvd_transformed = trunc.U * trunc.singular_values_
# print(x_tsvd_transformed)
# print("tsvd Explained Variance")
# print(np.var(x_tsvd_transformed, axis=0))
#
# U, Sigma, VT = svds(X, k=2, tol=0)
# Sigma = Sigma[::-1]
# U, VT = svd_flip(U[:, ::-1], VT[::-1])
# print("\n")
# print("Sklearn U matrix")
# print(U)
# print("Sklearn V^T")
# print(VT)
# print("Sklearn Sigma")
# print(Sigma)
# print("Sklearn U * Sigma")
# X_transformed = U * Sigma
# print(X_transformed)
# print("sklearn Explained Variance")
# print(np.var(X_transformed, axis=0))