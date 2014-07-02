function pogs_time = svm(m, n, params)
%SVM

if nargin == 2
  params = [];
end

% Generate data.
rng(0, 'twister')

lambda = 1.0;
N = m / 2;

x = 1 / n * [randn(N, n) + ones(N, n); randn(N, n) - ones(N, n)];
y = [ones(N, 1); -ones(N, 1)];
A = [(-y * ones(1, n)) .* x, -y];

f.h = kMaxPos;
f.b = -1;
f.c = lambda;
g.h = [kSquare(n); 0];

% Solve.
tic
pogs(A, f, g, params);
pogs_time = toc;

end

