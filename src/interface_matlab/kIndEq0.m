function x = kIndEq0(n)
%KINDEQ0 Function definition: h(x) = I(x = 0).

x = 5;

if nargin == 1
  x = x * ones(n, 1);
end