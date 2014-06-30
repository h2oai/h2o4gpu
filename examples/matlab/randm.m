function pogs_time = randm()
%randm

m = 100;
n = 2;
params = [];

% Load function definitions.
pogs_defs

% Generate data.
rng(0, 'twister');

A = rand(m, n);

% f.h = kExp;
% g.h = kSquare;

% f.h = kRecipr;
% g.h = kSquare;

f.h = kEntr;
g.h = kSquare;

% Solve
tic
pogs(A, f, g, params);
pogs_time = toc;

end
