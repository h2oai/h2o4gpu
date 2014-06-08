% Definitions
kAbs      = 0; kEntr     = 1; kExp     = 2; kHuber = 3;   kIdentity = 4; 
kIndBox01 = 5; kIndEq0   = 6; kIndGe0  = 7; kIndLe0 = 8;  kLogistic = 9;
kMaxNeg0 = 10; kMaxPos0 = 11; kNegLog = 12; kRecipr = 13; kSquare  = 14;
kZero    = 15;

pogs_setup

%% Lasso Example

% Setup
n = 200;
m = 2000;

rng(0, 'twister');

A = 1 / n * rand(m, n);
b = A * ((rand(n, 1) > 0.8) .* randn(n, 1)) + 0.1 * randn(m, 1);
lambda = 1e-2;

f.h = kSquare;
f.b = b;
g.h = kAbs;
g.c = lambda;

% Solve
tic
x_pogs = pogs(A, f, g);
pogs_time = toc;

tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(1 / 2 * sum_square_abs(A * x_cvx - b) + lambda * norm(x_cvx, 1))
cvx_end
cvx_time = toc;

fprintf('pogs_optval: %e, pogs_time: %e\n', ...
        1 / 2 * norm(A * x_pogs - b) ^ 2 + lambda * norm(x_pogs, 1), pogs_time);
fprintf('cvx_optval:  %e, cvx_time:  %e\n', ...
        1 / 2 * norm(A * x_cvx - b) ^ 2 + lambda * norm(x_cvx, 1), cvx_time);

%% Non-Negative Least Squares Example
rng(0, 'twister')

n_half = floor(0.9 * n);
A = 2 / n * rand(m, n);
b = A * [ones(n_half, 1); -ones(n - n_half, 1)] + 0.1 * randn(m, 1);

f.h = kSquare;
f.b = b;
g.h = kIndGe0;

% Solve
tic
x_pogs = pogs(A, f, g);
pogs_time = toc;

tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(1 / 2 * sum_square_abs(A * x_cvx - b))
  subject to
    x_cvx >= 0;
cvx_end
cvx_time = toc;

fprintf('pogs_optval: %e, pogs_time: %e\n', ...
        1 / 2 * norm(A * x_pogs - b) ^ 2, pogs_time);
fprintf('cvx_optval:  %e, cvx_time:  %e\n', ...
        1 / 2 * norm(A * x_cvx - b) ^ 2, cvx_time);

