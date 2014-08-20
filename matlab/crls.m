function [x,inform,resvec,lsvec] = crls(A,b,shift,tol,maxit,quiet)

% CRLS Conjugate-Residual Method for LS problems.
% [X,INFORM,RESVEC,LSVEC] = CRLS(A,B,SHIFT,TOL,MAXIT,QUIET) attempts to 
% solve min ||AX - B|| for X, where A is square or rectangular.
%
% On exit,
% INFORM = 1   if X solves AX = B to within TOL (e.g. TOL = 1e-8),
%              meaning the residual R = B - A*X is sufficiently small.
% INFORM = 2   if X solves the singular least-squares problem min ||AX-B||
%              to within TOL, meaning A'*R is sufficiently small, even
%              though R is not.
% INFORM = 3   if MAXIT iterations were performed before convergence.
% INFORM = 4   A seems to be singular.

% Authors:     David Fong and Michael Saunders, ICME, Stanford University.
% 21 Nov 2011: First version derived from cr.m.
% 20 Aug 2014: Added shift and quiet arguments.

  inform = 3;
  n      = size(A,2);
  x      = zeros(n,1);
  r      = b;
  s      = A'*r;                       % s   = A'r
  w      = A*s;                        % w   = As
  rho    = norm(w)^2 + shift*norm(s)^2;
  p      = s;
  q      = w;

  bnorm  = norm(b);
  snorm  = norm(s);
  Anorm  = snorm/bnorm;
  resvec = [bnorm; zeros(maxit,1)];    % Preallocate vector
  lsvec  = [snorm; zeros(maxit,1)];

  if ~quiet
    fprintf('\n\n    Itn     x(1)        Compatible')
    fprintf('  Least squares  norm(A)    norm(x)')
  end

  for itn = 1:maxit
    v     = A'*q + shift*p;            % v = A'q
    pnorm = norm(p);
    vnorm = norm(v);

    if vnorm <= Anorm*pnorm*eps        % A seems to be singular
       inform = 4;
       break
    end

    alpha = rho/vnorm^2;
    x     = x + alpha*p;
    r     = r - alpha*q;
    s     = s - alpha*v;
    w     = A*s;                       % w = As
   
    snorm = norm(s);
    xnorm = norm(x);
    rnorm = sqrt(norm(r)^2 + shift*xnorm^2);
    wnorm = sqrt(norm(w)^2 + shift*snorm^2);
    Anorm = max(Anorm,snorm/rnorm);
    resvec(itn+1) = rnorm;
    lsvec (itn+1) = snorm;

    test1 = rnorm/(Anorm*xnorm + bnorm);
    test2 = snorm/(Anorm*rnorm + 1e-99);

    if ~quiet
       str1 = sprintf('%6g %12.5e %13.3e', itn,x(1),test1);
       str2 = sprintf('%13.1e %10.1e %10.1e', test2,Anorm,xnorm);
       str  = [str1 str2];
      fprintf('\n %s', str)
    end
    
    if test1 <= tol                    % We have solved Ax = b
       inform = 1;
       break
    end
    if test2 <= tol                    % We have solved min ||Ax - b||
       inform = 2;
       break
    end

    rhoold = rho;
    rho    = wnorm^2;
    beta   = rho/rhoold;
    p      = s + beta*p;             
    q      = w + beta*q;               % q = Ap
  end

  resvec = resvec(1:itn+1);
  lsvec  =  lsvec(1:itn+1);
  if ~quiet
    disp(' ')
  end
end
