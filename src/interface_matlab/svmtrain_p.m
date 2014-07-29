function svmstruct = svmtrain_p(X, y, C, lambda)
%SVMTRAIN_P Use POGS to solve svm primal problem
%   Solves the probem
%
%     minimize    ||w||_2 + C \sum (1 - y_i [x_i^T 1] [w; b])_+ + lambda ||w||_1
%
%   svmstruct = svmtrain_p(X, y)
%   svmstruct = svmtrain_p(X, y, C, lambda)
%
%   Optional Inputs: C, lambda
%
%   Inputs:
%   X         - Data matrix where each row corresponds to an observation.
%
%   y         - Matrix of {-1, 1} values corresponding to the classes.
%
%   Optional Inputs
%   C         - Soft margin parameter.
%
%   lambda    - Sparsity regularizer.
%
%   Outputs:
%   svmstruct - Use as input to SVMCLASSIFY_P.
%
%   Example:
%       % Generate data
%       N = 100; N_test = 20; n = 2;
%       X = [randn(N, n) + 2 * ones(N, n); randn(N, n)];
%       y = [ones(N, 1); -ones(N, 1)];
%       % Train and test
%       svmstruct = svmtrain_p(X, y, 100);
%       X_test = [randn(N_test, n) + 2 * ones(N_test, n); randn(N_test, n)];
%       y_test = [ones(N_test, 1); -ones(N_test, 1)];
%       y_pred = svmclassify_p(X_test, svmstruct);
%       % Plot
%       xx = linspace(min(X_test(:, 1)), max(X_test(:, 1)));
%       plot(X_test(1:N_test, 1), X_test(1:N_test, 2), 'o',  ...
%            X_test(N_test+1:2*N_test, 1), X_test(N_test+1:2*N_test, 2), 'x', ...
%            xx, -(svmstruct.b + svmstruct.w(1) * xx) / svmstruct.w(2))
%       fprintf('Error %e\n', mean(y_test ~= y_pred))
%      
%
%   See also SVMCLASSIFY_P
%

if nargin < 3
  C = 1;
end
if nargin < 4
  lambda = 0;
end

[m, n] = size(X);

A = [X ones(m, 1)];
f.h = kMaxPos0;
f.a = -y;
f.b = -1;
f.c = C;
g.h = [kAbs(n); kZero];
g.c = lambda;
g.e = [ones(n, 1); kZero];

x = pogs(A, f, g);

svmstruct.w = x(1:n);
svmstruct.b = x(n + 1);

end
