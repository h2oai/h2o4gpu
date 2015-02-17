%RUN Runs all examples.

% Build interface
cd([pwd '/../../src/interface_matlab'])
pogs_setup -gpu % Alternatively `pogs_setup -gpu`
cd([pwd '/../../examples/matlab/'])
addpath([pwd '/../../src/interface_matlab'])


%% Run examples
fprintf('\nBasis Pursuit\n')
t = basis_pursuit(2000, 20000);
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nEntropy\n')
t = entropy(200, 2000);
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nHuber\n')
t = huber_fit(2000, 200);
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nLasso\n')
t = lasso(100, 1000);
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nLasso Path\n')
t = lasso_path(100, 1000);
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nLinear Program in Equality Form\n')
t = lp_eq(200, 1000, struct('rho', 100, 'max_iter', 10000));
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nLinear Program in Inequality Form\n')
t = lp_ineq(1000, 200, struct('rho', 1e-1, 'max_iter', 10000));
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nLogistic Regression\n')
t = logistic(1000, 200);
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nNon-Negative Least Squares\n')
t = nonneg_l2(1000, 200);
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nPortfolio Optimization\n')
t = portfolio(200, 2000);
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nPiecewise Linear Fitting\n')
t = pwl(2000, 200);
fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nSupport Vector Machine\n')
t = lasso(2000, 200);
fprintf('Solver Time: %e sec\n', t)

