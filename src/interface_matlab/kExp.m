function x = kExp(n)
%KEXP Function definition: h(x) = exp(x).

x = 1;

if nargin == 1
  x = x * ones(n, 1);
end