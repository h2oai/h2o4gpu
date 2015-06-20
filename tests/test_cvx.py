import cvxpy as cvx
import numpy as np

# SOC test
x = cvx.Variable(2)

c = np.matrix([1., 0.])
A = np.matrix([[-1., -1.],
               [ 0.,  1.]])

obj = cvx.Minimize(c * x)
constraints = [cvx.norm(A * x) <= 4.]
prob = cvx.Problem(obj, constraints)

result = prob.solve()
print x.value
print c * x.value

