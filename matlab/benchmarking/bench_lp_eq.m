function [n_iter, err, max_violation] = bench_lp_eq(dims, rho, params)

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
  A = 4 / n * rand(m, n);
  b = A * rand(n, 1);
  c = rand(n, 1);

  % Declare proximal operators.
  g_prox = @(x, rho) max(x, 0);
  f_prox = @(x, rho) [b; x(end) - 1 ./ rho(end)];
  obj_fn = @(x, y) y(end);
  
  % Solve using CVX to get optimal solution.
  if params.cmp_cvx
    cvx_begin quiet
      variable x_cvx(n)
      minimize(c' * x_cvx);
      subject to
        A * x_cvx == b;
        x_cvx >= 0;
    cvx_end
  else
    x_cvx = nan(n,1);
    cvx_optval = nan;
  end

  % Use factorization caching.
  factors = [];

  for j = 1:n_rho
    params.rho = rho(j);

    [x_pogs, ~, factors, n_it] = pogs(f_prox, g_prox, obj_fn, [A; c'], ...
                                      params, factors);

    max_violation(j, i) = ...
        abs(max([abs(b - A * x_pogs); max(-x_pogs, 0)])) / norm(x_pogs);
    n_iter(j, i) = n_it;
    err(j, i) = max(1e-6, (obj_fn(x_pogs, [A; c'] * x_pogs) - cvx_optval) / ...
        abs(cvx_optval));
  end
end

