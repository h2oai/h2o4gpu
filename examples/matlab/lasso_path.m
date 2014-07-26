function pogs_time = lasso_path(m, n, params)
%LASSO_PATH

if nargin == 2
  params = [];
end

% Generate data.
rng(0, 'twister');

A = rand(m, n);
b = A * ((rand(n, 1) > 0.8) .* randn(n, 1) /  n) + 0.1 * randn(m, 1);

N = 100;

lambda_max = norm(A' * b, inf);
lambdas = exp(linspace(log(lambda_max), log(lambda_max * 1e-2), N));
f = repmat(struct, N, 1);
g = repmat(struct, N, 1);
for i = 1:N
  f(i).h = kSquare;
  f(i).b = b;
  g(i).h = kAbs;
  g(i).c = lambdas(i);
end

% Solve
tic
pogs(A, f, g, params);
pogs_time = toc;

end

