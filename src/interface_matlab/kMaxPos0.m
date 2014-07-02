function x = kMaxPos0(n)
%KMAXPOS0 Function definition: h(x) = max(0, x).

x = 10;

if nargin == 1
  x = x * ones(n, 1);
end