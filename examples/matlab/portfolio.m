function pogs_time = portfolio(m, n, params)
%PORTFOLIO

if nargin == 2
  params = [];
end

% Generate data.
rng(0, 'twister');

A = [randn(n, m) ones(n, 1)]';
d = rand(n, 1);
r = -rand(n, 1);
gamma = 1;

f.h = kSquare;
f.c = gamma;
g.h = kIndGe0;
g.d = r;
g.e = gamma * d;

% Solve
tic
pogs(A, f, g, params);
pogs_time = toc;

end

