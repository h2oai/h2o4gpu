%RUN Runs all examples.

% Build interface
cd([pwd '/../../src/interface_matlab'])
pogs_setup -gpu % Alternatively `pogs_setup gpu`
cd([pwd '/../../examples/matlab/'])
addpath([pwd '/../../src/interface_matlab'])

%% Run examples

% fprintf('\nLasso\n')
% t = lasso_sp(100, 1000, 1e4, struct('abs_tol', 1e-5, 'rel_tol', 1e-4, 'max_iter', 1001))
% fprintf('Solver Time: %e sec\n', t)
%%
fprintf('\nLP-Equality\n')
[t1, t2] = lp_eq_sp(200, 1000, 1e4, struct('abs_tol', 1e-4, 'rel_tol', 1e-3, 'max_iter', 2001))
fprintf('Solver Time: %e sec\n', t1)