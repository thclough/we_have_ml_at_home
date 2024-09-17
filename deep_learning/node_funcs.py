import numpy as np

## ACTIVATION FUNCTION CLASSES
class Sigmoid:
    
    @staticmethod
    def forward(x):
        """compute the sigmoid for the input x

        Args:
            x: A numpy float array

        Returns: 
            A numpy float array containing the sigmoid of the input
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def backward(x):
        """Gradient of the sigmoid function evaluated at a vector x
        calculates Jacobian of "a" with respect to "z" in NN
        
        Args:
            x (vector) : input vector
        
        Returns:
            (vector) : Jacobian wrt x
        
        """
        return Sigmoid.forward(x) * (1 - Sigmoid.forward(x))
    
class HardSigmoid:
    
    @staticmethod
    def forward(x):
        """compute the sigmoid for the input x
        """
        return np.maximum(0, np.minimum(1, (x+1)/2))
    
    @staticmethod
    def backward(x):
        """
        
        """
        return ((-2 <= x) & (x <= 2)) * .5

class Identity:

    @staticmethod
    def forward(x):
        """Identity function, the output is the input
    
        Args:
            x (numeric, or vector) : the input

        Returns:
            x (numeric, or vector) : the input
        """
        return x
    
    @staticmethod
    def backward(x):
        """Calculate Jacobian of the identity function wrt x
        
        Args:
            x (vector) : input vector

        Returns:
            (vector) : Jacobian wrt x, just a vector of ones
        """
        return np.ones(x.shape)

class ReLU:

    name = "Rectified Linear Unit"

    @staticmethod
    def forward(x):
        """Rectified linear unit: max(0,input)
        
        Args:
            x (vector): input vector of ReLU
        
        Returns:
            (vector) : ReLU output
        
        """
        return np.maximum(0,x)
    
    @staticmethod
    def backward(x):
        """Jacobian of ReLU
        
        Args:
            x (vector) : input of ReLU
        
        Returns:
            (vector) : Jacobian of the ReLU
        """
        return x >= 0
    
class LeakyReLU:

    @staticmethod
    def forward(x):
        """Leaky rectified linear unit: max(0.1*input,input)
        
        Args:
            x (vector): input vector of Leaky ReLU
        
        Returns:
            (vector) : Leaky ReLU output
        
        """
        return np.maximum(0.1*x,x)
    
    @staticmethod
    def backward(x):
        """Jacobian of leaky ReLU
        
        Args:
            x (vector) : input of Leaky ReLU
        
        Returns:
            (vector) : Jacobian of the Leaky ReLU
        """
        return np.where(x < 0, 0.1, 1)
    
class TanH:

    @staticmethod
    def forward(x):
        """Hyperbolic tangent : (e^x - e^-x) / (e^x + e^-x)
        
        Args:
            x (vector): input vector of tanh
        
        Returns:
            (vector) : tanh output
        
        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def backward(x):
        """Jacobian of hyperbolic tangent
        
        Args:
            x (vector) : input of tanh
        
        Returns:
            (vector) : Jacobian of tanh given input
        """
        return 1 - TanH.forward(x)**2
    
# SOFTMAX

class Softmax:

    @staticmethod
    def forward(x):
        # x_shifted = x - np.max(x, axis=1, keepdims=True)
    
        # exp_mat = np.exp(x_shifted)

        # sm_mat = exp_mat / np.sum(exp_mat, axis=1, keepdims=True)

        # return sm_mat
        x = x - np.max(x,axis=1)[:,np.newaxis]
        exp = np.exp(x)
        s = exp / np.sum(exp,axis=1)[:,np.newaxis]
        return s

    @staticmethod
    def backward(x):
        return Softmax.forward(x) * (1-Softmax.forward(x))

## LOSS FUNCTIONS

### CLASSIFICATION

class BCE: 
    """binary cross entropy for binary classification"""
    name = "Binary Cross Entropy"
    @staticmethod
    def forward(y_pred,y_true):
        """
        Args:
            y_pred (vector): output probabilities
            y_true (vector): ground_truth labels

        Returns:
            (vector) : binary cross entropy loss
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    
    @staticmethod
    def backward(y_pred, y_true):
        """Compute the gradient with respect to linear combo (z's)
        
        Args: 
            y_pred (vector): output probabilities
            y_true (vector): ground_truth labels

        Returns:
            (vector) : Jacobian wrt loss
        """
        return y_pred - y_true

class WeightedBCE(BCE):
    """For imbalanced dataset"""
    name = "Weighted Binary Cross Entropy"
    
    def __init__(self, prop_pos):
        """
        Args:
            prop_pos (float) : proportion of examples that are positive
        """
        self._weight_pos = (1-prop_pos) / prop_pos

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (vector) : output probabilities
            y_true (vector) : ground truth labels

        Returns:
            (vector) : binary cross entropy
        """
        weight_vector = self._get_weight_vector(y_true)
        return weight_vector * super().forward(y_pred, y_true)

    def backward(self, y_pred, y_true):
        """

        Args: 
            y_pred (vector): output probabilities
            y_true (vector): ground truth labels

        Returns:
            (vector) : Jacobian wrt linear combination
        """
        weight_vector = self._get_weight_vector(y_true)
        return weight_vector * super().backward(y_pred, y_true)
    
    def _get_weight_vector(self, y_true):
        """Obtain the correct weight vector for forward and backward
        
        Args:
            y_true (vector) : ground truth labels

        Returns:
            (vector) : weight vector with same dimensions as y_true
        """
        return self._weight_pos * y_true + (1-y_true)

class CE:
    name = "Cross Entropy"
    """regular cross entropy for multi-class classification"""
    @staticmethod
    def forward(y_pred, y_true):
        """y_true is a sparse array"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred), axis=1)
    
    @staticmethod
    def backward(y_pred, y_true):
        return y_pred - y_true 
    
### REGRESSION

class RegressionLoss:
    pass


class MSE(RegressionLoss): 
    """Mean Squared Error for Regression"""
    name = "Mean Squared Error"

    @staticmethod
    def forward(y_pred,y_true):
        """
        Args:
            y_pred (vector): output values
            y_true (vector): ground_truth values

        Returns:
            (vector) : MSE loss
        """
        return ((y_pred-y_true) ** 2)/2
    
    @staticmethod
    def backward(y_pred, y_true):
        """Compute the gradient with respect to linear combo (z's)
        
        Args: 
            y_pred (vector): output values
            y_true (vector): ground_truth labels

        Returns:
            (vector) : Jacobian wrt loss
        """

        return y_pred - y_true