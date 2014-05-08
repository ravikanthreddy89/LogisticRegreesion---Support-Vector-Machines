function [label] = mlrPredict(W, X)
% blrObjFunction predicts the label of data given the data and parameter W
% of multi-class Logistic Regression
%
% Input:
% W: the matrix of weight of size (D + 1) x 10
% X: the data matrix of size N x D
%
% Output: 
% label: vector of size N x 1 representing the predicted label of
%        corresponding feature vector given in data matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%label = zeros(size(X, 1), 1); % dummy return


intercept=ones(size(X,1),1);
X=horzcat(intercept,X);
%output=sigmoid(X*W);

 a=(X*W);    
    
 y=exp(a);
 y_n=bsxfun(@rdivide, y, sum(y,2));


[~ , label]=max(y_n,[],2);

end

