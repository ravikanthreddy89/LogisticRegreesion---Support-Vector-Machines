function [W] = mlrNewtonRaphsonLearn(initial_W, X, T, n_iter)
%mlrNewtonRaphsonLearn learns the weight vector of multi-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_W: matrix of size (D+1) x 10 represents the initial weight matrix 
%            for iterative method
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% W: matrix of size (D+1) x 10, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = zeros(size(X, 2) + 1, 10); % dummy return
w_new=zeros(size(X, 2) + 1, 10);

mat_1=ones(size(X,1),1);
X=horzcat(mat_1,X);%add intercept to the X


d=size(X,2);
classes=size(T,2);
n=size(X,1);

H=zeros(d*classes, d*classes);
I=eye(classes);

w_old=initial_W(:);

for i= 1 : n_iter
    a=(X*reshape(w_old, d,classes));    
    
     y=exp(a);
     y_n=bsxfun(@rdivide, y, sum(y,2));

     error_grad=X'*(y_n-T);
     
     b=logsumexp(a,2);

     error1=-1* sum(sum(a.*T));
     error2=sum(b);

     error=error1+error2;
     
     row=1;
     col=1;
     for a= 1: classes
         y_na= y_n(:, a);
         for b= 1:classes
             y_nb=y_n(:,b);
             I_local= repmat(I(a,b), n, 1);
            % R=repmat(((I_local-y_nb).*y_na), 1,d);
            % H(row:row+d-1, col:col+d-1)= -1*(X'*(X.*R));
            % R=repmat(((y_nb-I_local).*y_na)',size(X,2),1);
            % H(row:row+d-1, col:col+d-1)= (X'.*R)*X;
              R=diag(((I_local-y_nb).*y_na));
              H(row:row+d-1, col:col+d-1)= X'*R*X;
              col=col+d;
         end
         row=row+d;
         col=1;
     end
      
     temp=double(pinv(H));
     w_new=w_old-temp*error_grad(:);%update rule
     w_old=w_new;
 end
W=reshape(w_new, d, classes);
end

