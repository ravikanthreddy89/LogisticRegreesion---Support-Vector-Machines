LogisticRegreesion---Support-Vector-Machines
============================================

Handwritten digit classification using logistic regression and support vector machines

Following methods are used for classification of MNIST handwritten digit data

1) Multiclass logistic regression
    -> using gradient descent
    -> using Newton Raphson's method by computing the Hessian matrix
    
   I followed the text in Bishop's book to implement the multi class logistic regression. 
   
   NOTE : In Newton Raphson's method where we compute the Hessian might throw out of memory error. Make sure you 
   change the default memory setting of you Matlab installation/setup before running.
    
2) Support Vector machines
    -> libsvm is being used, no explicit code.
