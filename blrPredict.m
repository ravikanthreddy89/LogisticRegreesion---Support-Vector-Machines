function [label] = blrPredict(W, X)
% blrObjFunction predicts the label of data given the data and parameter W
% of Logistic Regression
%
% Input:
% W: the matrix of weight of size (D + 1) x 10. Each column is the weight
%    vector of a Logistic Regression classifier.
% X: the data matrix of size N x D
%
% Output: 
% label: vector of size N x 1 representing the predicted label of
%        corresponding feature vector given in data matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%label = zeros(size(X, 1), 1); % dummy return

%add the intercept
intercept=ones(size(X,1),1);
X=horzcat(intercept,X);
output=sigmoid(X*W);

[~ , label]=max(output,[],2);
end

