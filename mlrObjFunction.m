function [error, error_grad] = mlrObjFunction(W, X, T)
% mlrObjFunction computes multi-class Logistic Regression error function 
% and its gradient.
%
% Input:
% W: the vector of size ((D + 1) * 10) x 1. Later on, it will reshape into
%    matrix of size (D + 1) x 10
% X: the data matrix of size N x D
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size ((D+1) * 10) x 1 representing the gradient 
%             of error function


W = reshape(W, size(X, 2) + 1, size(T, 2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%error = 0; % dummy return
%error_grad = zeros(size(X, 2) + 1, 1); % dummy return

%add intercept to the data

intercept=ones(size(X,1),1);
X=horzcat(intercept,X);
a=X*W;


b=logsumexp(a,2);

error1=-1* sum(sum(a.*T));
error2=sum(b);

error=error1+error2;

y=exp(a);
y_n=bsxfun(@rdivide, y, sum(y,2));

temp=X'*(y_n-T);
error_grad=temp(:);

end
