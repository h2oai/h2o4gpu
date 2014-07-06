function pogs_time = huber_fit(m, n, params)
%HUBER_FIT

if nargin == 2
  params = [];
  params.rho = 1e2;
end

% Generate data.
rng(0, 'twister');

A = randn(m, n);
p = rand(m, 1);
b = randn(m, 1) .* (p <= 0.95) + 10 * rand(m, 1) .* (p > 0.95);

f.h = kHuber;
f.b = b;
g.h = kZero;

% Solve
tic
pogs(A, f, g, params);
pogs_time = toc;

end

