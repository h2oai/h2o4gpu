function x = kHuber(n)
%KHUBER Function definition: h(x) = huber(x).

x = 2;

if nargin == 1
  x = x * ones(n, 1);
end