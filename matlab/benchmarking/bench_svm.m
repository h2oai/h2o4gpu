function [n_iter, err, max_violation] = bench_svm(dims, rho, params)

n_dim = size(dims, 1);
n_rho = length(rho);

n_iter = nan(n_rho, n_dim);
err = nan(n_rho, n_dim);
max_violation = nan(n_rho, n_dim);

for i = 1:n_dim
  m = dims(i, 1);
  n = dims(i, 2);
  
  lambda = 1.0;
  N = m / 2;

  % Initialize rng.
  rng(0, 'twister')

  % Generate data.
  x = 1 / n * [randn(N, n) + ones(N, n); randn(N, n) - ones(N, n)];
  y = [ones(N, 1); -ones(N, 1)];
  A = [(-y * ones(1, n)) .* x, -y];

  % Declare proximal operators.
  f_prox = @(x, rho) max(0, x + 1 - lambda ./ rho) + min(0, x + 1) - 1;
  g_prox = @(x, rho) [rho(1:end - 1) .* x(1:end - 1) ./ (1 + rho(1:end - 1)); x(end)];
  obj_fn = @(x, y) 1 / 2 * norm(x(1:n)) ^ 2 + lambda * sum(max(0, y + 1));
  
  % Solve using CVX to get optimal solution.
  if params.cmp_cvx
    cvx_begin quiet
      variable x_cvx(n + 1)
      minimize(1 / 2 * x_cvx(1:n)' * x_cvx(1:n) + ...
          lambda * sum(max(0, A * x_cvx + 1)));
    cvx_end
  else
    x_cvx = nan(n + 1,1);
    cvx_optval = nan;
  end

  % Use factorization caching.
  factors = [];

  for j = 1:n_rho
    params.rho = rho(j);

    [x_pogs, ~, factors, n_it] = pogs(f_prox, g_prox, obj_fn, A, ...
                                      params, factors);

    max_violation(j, i) = 0;
    n_iter(j, i) = n_it;
    err(j, i) = max(1e-6, (obj_fn(x_pogs, A * x_pogs) - cvx_optval) / ...
        abs(cvx_optval));
  end
end
