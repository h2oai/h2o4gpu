function x = kIndBox01(n)
%KINDBOX01 Function definition: h(x) = I(0 <= x <= 1).

x = 4;

if nargin == 1
  x = x * ones(n, 1);
end