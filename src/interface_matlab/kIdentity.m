function x = kIdentity(n)
%KIDENTITY Function definition: h(x) = x.

x = 3;

if nargin == 1
  x = x * ones(n, 1);
end