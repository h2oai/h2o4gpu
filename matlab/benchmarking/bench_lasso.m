function [n_iter, err, max_violation] = bench_lasso(dims, rho, params)

n_dim = size(dims, 1);
n_rho = length(rho);

n_iter = nan(n_rho, n_dim);
err = nan(n_rho, n_dim);
max_violation = nan(n_rho, n_dim);

for i = 1:n_dim
  m = dims(i, 1);
  n = dims(i, 2);

  % Initialize rng.
  rng(0, 'twister')

  % Generate data.
  A = 5 / n * randn(m, n);
  b = A * ((rand(n, 1) > 0.8) .* randn(n, 1)) + 0.5 * randn(m, 1);
  lambda = 0.4 + 1e-4 * m;

  % Declare proximal operators.
  g_prox = @(x, rho) max(x - lambda ./ rho, 0) - max(-x - lambda ./ rho, 0);
  f_prox = @(x, rho) (rho .* x + b) ./ (1 + rho);
  obj_fn = @(x, y) 1 / 2 * norm(y - b) ^ 2 + lambda * norm(x, 1);
  
  % Solve using CVX to get optimal solution.
  if params.cmp_cvx
    cvx_begin quiet
      variables x_cvx(n) y(m)
      minimize(1 / 2 * (y - b)' * (y - b) + lambda * norm(x_cvx, 1));
      subject to
        A * x_cvx == y;
    cvx_end
  else
    x_cvx = nan(n,1);
    cvx_optval = nan;
  end

  % Use factorization caching.
  factors = [];

  for j = 1:n_rho
    params.rho = rho(j);
    params.b = b;
    params.lambda = lambda;

    [x_pogs, ~, factors, n_it] = pogs(f_prox, g_prox, obj_fn, A, ...
                                      params, factors);

    max_violation(j, i) = 0;
    n_iter(j, i) = n_it;
    err(j, i) = max(1e-6, (obj_fn(x_pogs, A * x_pogs) - cvx_optval) / ...
        abs(cvx_optval));
  end
end

