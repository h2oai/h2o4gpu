function y_new = svmclassify_p(svmstruct, X)
%SVMCLASSIFY_P Classification function for SVMTRAIN_P
%
%   y_new = svmclassify_p(svmstruct, X)
%
%   Inputs:
%   svmstruct - Output of SVMTRAIN_P
%
%   X         - Data matrix for prediction.
%
%   Outputs:
%   y_new     - Predicted class.
%
%   See also SVMTRAIN_P

y_new = sign(X * svmstruct.w + repmat(svmstruct.b, size(X, 1), 1));

end
