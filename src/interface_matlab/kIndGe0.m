function x = kIndGe0(n)
%KINDGE0 Function definition: h(x) = I(x >= 0).

x = 6;

if nargin == 1
  x = x * ones(n, 1);
end