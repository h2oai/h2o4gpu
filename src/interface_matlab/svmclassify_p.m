function y_new = svmclassify_p(X, svmstruct)
%SVMCLASSIFY_P Classification function for SVMTRAIN_P
%
%   y_new = svmclassify_p(X, svmstruct)
%
%   Inputs:
%   X         - Data matrix for prediction.
%
%   svmstruct - Output of SVMTRAIN_P
%
%   Outputs:
%   y_new     - Predicted class.
%
%   See also SVMTRAIN_P

y_new = sign(X * svmstruct.w + svmstruct.b);

end
