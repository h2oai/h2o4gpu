%RUN Runs all examples.

% Build interface
cd([pwd '/../../src/interface_matlab'])
% % Use one of the following to set up
h2ogpuml_setup 
% h2ogpuml_setup -omp
% h2ogpuml_setup -gpu
% h2ogpuml_setup -gpu -cuda_lib /usr/local/cuda/lib -cuda_bin /usr/local/cuda/bin
cd([pwd '/../../examples/matlab/'])
addpath([pwd '/../../src/interface_matlab'])

%% Run examples
fprintf('\nBasis Pursuit\n')
t = basis_pursuit(200, 2000);
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
fprintf('\nLinear Program in Cone Form\n')
t = lp_cone(1000, 200);
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
t = svm(2000, 200);
fprintf('Solver Time: %e sec\n', t)

