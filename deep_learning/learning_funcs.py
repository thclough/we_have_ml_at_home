## Learning rate schedulers

class Decay:

    """
    Attributes:
        learning_rate_0 (float) : initial learning rate
        decay_rate (float): hyperbolic decay rate 
    """

    def __init__(self, learning_rate_0:float, decay_rate:float):
        self._learning_rate_0 = learning_rate_0
        self._decay_rate = decay_rate

class HyperbolicDecay(Decay):
    
    def __init__(self, learning_rate_0:float, decay_rate:float=.9):
        if decay_rate <= 0:
            raise Exception("Decay rate must be positive")
        super().__init__(learning_rate_0, decay_rate)

    def get_learning_rate(self, epoch_num):
        """ 
        Args:
            epoch_num (int) : epoch number zero-indexed
        
        Returns:
            learning_rate (float) : learning rate for the epoch
        """

        learning_rate = self._learning_rate_0/(1+self._decay_rate * epoch_num)
    
        return learning_rate

class ExpoDecay(Decay):

    def __init__(self, learning_rate_0:float, decay_rate:float=.9):
        if decay_rate <=0 or decay_rate >= 1:
            raise Exception("Decay rate must be between 0 and 1 exclusive")
        super().__init__(learning_rate_0, decay_rate)

    def get_learning_rate(self, epoch_num):
        """ 
        Args:
            epoch_num (int) : epoch number zero-indexed
        
        Returns:
            learning_rate (float) : learning rate for the epoch
        """

        learning_rate = self._learning_rate_0 * self._decay_rate ** epoch_num

        return learning_rate

class ConstantRate:
    """
    Attributes:
        learning_rate_0 (float) : initial learning rate
    
    """
    def __init__(self, learning_rate_0:float=1):
        self._learning_rate_0 = learning_rate_0

    def get_learning_rate(self, epoch_num):
        """ 
        Args:
            epoch_num (int) : epoch number zero-indexed
        
        Returns:
            learning_rate (float) : learning rate for the epoch
        """

        return self._learning_rate_0

