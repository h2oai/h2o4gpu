function [pogs_time, cvx_time] = entropy(m, n, params, cvx_comp)
%ENTROPY

if nargin <= 2
  params = [];
end
if nargin <= 3
  cvx_comp = false;
end

cvx_time = nan;

% Generate data.
rng(0, 'twister');

F = randn(m, n) * sqrt(n);
x_p = rand(n, 1);
x_p = x_p / sum(x_p);
b = F * x_p + 0.5 * rand(m, 1);
A = [F; ones(1, n)];

f.h = [kIndLe0(m); kIndEq0];
f.b = [b; 1];
g.h = kNegEntr;

% Solve with pogs
As = single(A);
tic
[~, ~, ~, ~, status] = pogs(As, f, g, params);
pogs_time = toc;

if status > 0
  pogs_time = nan;
end

% Solve with CVX
if cvx_comp
  tic
  cvx_begin
    variables x(n)
    minimize(-sum(entr(x)))
    subject to
      F * x <= b;
      sum(x) == 1;
  cvx_end
  cvx_time = toc;
end

end
