from base import Model

import numpy as np


class LinearRegression(Model):


    def __init__(self):
        pass


    def loss(self, y, y_hat):
        """
        Mean Squared Error.
        """

        return 1/len(y) * np.sum( np.square(y - y_hat) )


    def gradient(self, X, y, w):
        """
        Gradient of MSE loss.
        """
        return 1 / len(y) * (-2*X.T.dot(y) + 2 * X.T.dot(X.dot(w)))
    

    def gradient_descent(
        self,
        X, y,
        w,
        alpha,
        iteration_n = 1000
    ):
        
        i=0
        error = self.loss(y, X.dot(w))
        
        while i < iteration_n:
            
            w = w - alpha * self.gradient(X,y,w)
            print(w)
            
            error = self.loss(y, X.dot(w))
            print(error)
            
            i+=1
        
        return w