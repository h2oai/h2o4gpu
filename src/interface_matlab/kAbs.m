function x = kAbs(n)
%KABS Function definition: h(x) = |x|.

x = 0;

if nargin == 1
  x = x * ones(n, 1);
end