function x = kLogistic(n)
%KLOGISTIC Function definition: h(x) = log(1 + e^x).

x = 8;

if nargin == 1
  x = x * ones(n, 1);
end