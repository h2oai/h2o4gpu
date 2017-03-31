%%POGS Generic graph projection splitting solver.
%   Solves problems in the form
%
%     minimize    f(y) + g(x),
%     subject to  y = Ax.
% 
%   where the f and g are separable and A is dense.
%
%   [x, y, l, optval] = pogs(A, f, g)
%   [x, y, l, optval] = pogs(A, f, g, params)
% 
%   Optional Inputs: params
%
%   Inputs:
%   A         - Matrix A corresponding to constraint y = Ax.
%
%   f         - f struct with fields h, a, b, c, d and e, each of which
%               must either be a vector of dimension m (resp. n) or a 
%               scalar. If a scalar is specified, then it is assumed that
%               the scalar should be repeated m (resp. n) times.
%               Corresponds to function f in objective. If f is an array
%               of structs, then POGS will solve each each problem in
%               succession (this will be much faster than solving the
%               problems individually, since POGS will perform
%               factorization caching).
%
%   g         - Same as f, except that it corresponds to function g in
%               objetive.
%
%   Optional Inputs:
%   params    - Structure of parameters, containing any of the following
%               fields:
%                 + abs_tol (default 1e-4): Absolute tolerance to which the
%                   problem should be solved.
%                 + rel_tol (default 1e-3): Relative tolerance to which the
%                   problem should be solved.
%                 + max_iter (default 1000): Maximum number of iterations
%                   that the solver should be run for.
%                 + rho (default 1.0): Penalty parameter for proximal
%                   operator.
%                 + adaptive_rho (default true): Change rho adaptively to
%                   speed up conversion.
%                 + equil (default true): Do equilibration.
%                 + quiet (default false): Set flag to true, to disable
%                   output to console.
%
%   Outputs:
%   x         - The partial solution x^\star to the optimization problem.
%
%   y         - The partial solution y^\star to the optimization problem.
%
%   l         - The dual variable corresponding to the constraint 
%               Ax - y = 0.
%   
%   optval    - The optimal value of the objective f(y^\star) + g(x^\star).
%
%   References: 
%   http://foges.github.io/pogs
%     Documentation for POGS -- C. Fougner
%
%   http://www.stanford.edu/~boyd/papers/block_splitting.html 
%     Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd
%
%   http://www.stanford.edu/~boyd/papers/admm_distr_stats.html
%     Distributed Optimization and Statistical Learning via the Alternating
%     Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu,
%     B. Peleato, and J. Eckstein
%
%   http://www.stanford.edu/~boyd/papers/prox_algs.html
%     Proximal Algorithms -- N. Parikh and S. Boyd
%
%   Author:
%   Christopher Fougner.
%
