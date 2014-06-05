function [n_iter, err, max_violation] = bench_lp_ineq(dims, rho, params)

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
  A = -[4 / n * rand(m - n, n); eye(n)];
  b = A * rand(n, 1) + 0.2 * rand(m, 1);
  c = rand(n, 1);

  % Declare proximal operators.
  g_prox = @(x, rho) x - c ./ rho;
  f_prox = @(x, rho) min(b, x);
  obj_fn = @(x, y) c' * x;
  
  % Solve using CVX to get optimal solution.
  if params.cmp_cvx
    cvx_begin quiet
      variable x_cvx(n)
      minimize(c' * x_cvx);
      subject to
        A * x_cvx <= b;
    cvx_end
  else
    x_cvx = nan(n,1);
    cvx_optval = nan;
  end

  % Use factorization caching.
  factors = [];

  for j = 1:n_rho
    params.rho = rho(j);

    [x_pogs, ~, factors, n_it] = pogs(f_prox, g_prox, obj_fn, A, ...
                                      params, factors);

    max_violation(j, i) = ...
        abs(min(min(b - A * x_pogs), 0)) / norm(b - A * x_pogs);
    n_iter(j, i) = n_it;
    err(j, i) = max(1e-6, (obj_fn(x_pogs, A * x_pogs) - cvx_optval) / ...
        abs(cvx_optval));
  end
end

