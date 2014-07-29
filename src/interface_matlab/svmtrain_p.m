function svmstruct = svmtrain_p(X, y, C, lambda, params)
%SVMTRAIN_P Use POGS to solve penalized linear svm primal problem
%   Solves the probem
%
%     minimize    (1/2)||w||_2 + C \sum (1 - y_i [x_i^T 1] [w; b])_+ + lambda ||w||_1
%
%   svmstruct = svmtrain_p(X, y)
%   svmstruct = svmtrain_p(X, y, C, lambda, params)
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
%   lambda    - Sparsity regularizer. Can be vector to solve for multiple
%               values of lambda.
% 
%   params    - Parameters to POGS.
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
%       xx = linspace(min(X_test(:, 1)), max(X_test(:, 1)))';
%       plot(X_test(1:N_test, 1), X_test(1:N_test, 2), 'o',  ...
%            X_test(N_test+1:2*N_test, 1), X_test(N_test+1:2*N_test, 2), 'x', ...
%            xx, -(svmstruct.b + xx * svmstruct.w(1)) / svmstruct.w(2))
%       fprintf('Error %e\n', mean(y_test ~= y_pred))
%
%   See also SVMCLASSIFY_P
%

if nargin < 3 || isempty(C)
  C = 1;
end
if nargin < 4 || isempty(lambda)
  lambda = 0;
end
if nargin < 5
  params = [];
end

if any(y ~= 1 & y ~= -1)
  error('All entries in y must be in {-1, 1}')
end

[m, n] = size(X);

A = [X ones(m, 1)];
f = repmat(struct('a', -y, 'b', -1, 'c', C, 'h', kMaxPos0), length(lambda), 1);
g = struct('c', num2cell(lambda), 'e', [ones(n, 1); kZero], 'h', [kAbs(n); kZero]);

x = pogs(A, f, g, params);
svmstruct.w = x(1:n, :);
svmstruct.b = x(n + 1, :);

end
