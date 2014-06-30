%RUN Runs all examples.

% Build interface
cd([pwd '/../../src/interfaces/'])
pogs_setup
% pogs_setup('gpu')
cd([pwd '/../../examples/matlab/'])
addpath([pwd '/../../src/interfaces'])

%% Run examples
fprintf('\nEntropy\n')
t = entropy(200, 2000);
fprintf('Solver Time: %e sec\n', t)

fprintf('\nLasso\n')
t = lasso(1000, 500);
fprintf('Solver Time: %e sec\n', t)

fprintf('\nLinear Program in Equality Form\n')
t = lasso(200, 1000);
fprintf('Solver Time: %e sec\n', t)

fprintf('\nLinear Program in Inequality Form\n')
t = lasso(1000, 200);
fprintf('Solver Time: %e sec\n', t)

fprintf('\nLogistic Regression\n')
t = logistic(1000, 200);
fprintf('Solver Time: %e sec\n', t)

fprintf('\nNon-Negative Least Squares\n')
t = lasso(1000, 200);
fprintf('Solver Time: %e sec\n', t)

fprintf('\nSupport Vector Machine\n')
t = lasso(1000, 200);
fprintf('Solver Time: %e sec\n', t)
