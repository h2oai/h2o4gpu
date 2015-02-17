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

F = randn(m, n) / n;
x_any = rand(n, 1); x_any = x_any / sum(x_any);
b = F * x_any + 0.1 * rand(m, 1);
A = [F; ones(1, n)];

f.h = [kIndLe0(m); kIndEq0];
f.b = [b; 1];
g.h = kNegEntr;

% Solve with pogs
A = single(A);
tic
pogs(A, f, g, params);
pogs_time = toc;

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
