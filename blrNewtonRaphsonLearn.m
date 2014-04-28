function [w] = blrNewtonRaphsonLearn(initial_w, X, t, n_iter)
%blrNewtonRaphsonLearn learns the weight vector of 2-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_w: vector of size (D+1) x 1 where D is the number of features in
%            feature vector
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% t: vector of size N x 1 where each entry is either 0 or 1 representing
%    the true label of corresponding feature vector.
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% w: vector of size (D+1) x 1, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = zeros(size(X, 2) + 1, 1); % dummy return
w_new=zeros(size(X, 2) + 1, 1);

%add the intercept to the data
mat_1=ones(size(X,1),1);
X=horzcat(mat_1,X);

%let the gamma be 1 

w_old=initial_w;

for i= 1 : n_iter
    y_n=sigmoid(X*w);    
    error_grad=X'*(y_n-t);%gradient
    
    temp=(mat_1-y_n).*y_n;
    %H=X'*B*X;%Hessian matrix
    %it is not working the diag() size = 50k X 50k
    %so lets use inbuilt hessian function 
    %oh wait we can't use it either.
    alpha=repmat(temp', size(X,2),1);
    
    H=(X'.*alpha)*X;
    
    w_new=w_old-pinv(H)*error_grad;%update rule
    
    w_old=w_new;
    
end
w=w_new;
end
