LogisticRegreesion---Support-Vector-Machines
============================================

Handwritten digit classification using logistic regression and support vector machines

The repo contains implementation of various models to classify handwritten digits. 

1) Model 1 : Binary logistic regression

    -> In this model ten one-vs-all classifier corresponding to each digit are built.
    
    -> During prediction the output of classifier with maximum value is considered. 

    
2) Model 2 : Multiclass logistic regression

    -> This model is an extension of one-vs-all classifier.
    
    -> In this case the classifier outputs the K probabilities each representing the chance of input belonging to 
       one of the K classes.
    
    
   I followed the text in Bishop's book (pg 205-210) to implement the multi class logistic regression. 
   
   NOTE : In Newton Raphson's method where we compute the Hessian might throw out of memory error. Make sure you 
   change the default memory setting of you Matlab installation/setup before running.
    
2) Support Vector machines

    -> libsvm is being used, no explicit code.
