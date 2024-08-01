# model structures to pass into model manager
# have their own forward prop, epoch, and back prop routines

import numpy as np
from deep_learning import nn_layers
from deep_learning import node_funcs
import joblib

class FeedMeForwardNN: 
    """Feed forward neural network"""

    def __init__(self):
        self.layers = []
        self.loss_layer = None
        self._last_output_layer = None # last layer added that changes output dimension

        self._has_fit = False
        self._has_dropout = False
    
    # STRUCTURE

    def add_layer(self, layer_object):

        if self.loss_layer is not None:
            raise Exception("Cannot add another layer after Loss layer")

        if len(self.layers) == 0:
            if isinstance(layer_object, nn_layers.Web):
                if layer_object.input_shape is None:
                    raise Exception("First layer input shape must be set in layer")
                
                layer_object.input_layer_flag = True
                self._last_output_layer = layer_object
            else:
                raise Exception("First layers must be a web layer")

        if len(self.layers) > 0:
            if isinstance(layer_object, (nn_layers.Web, nn_layers.Dropout, nn_layers.BatchNorm)):
                if layer_object.input_shape is not None:
                    if layer_object.input_shape != self._last_output_layer.output_shape:
                        raise Exception("Input shape must equal the output shape of the last web")
                else:
                    layer_object.input_shape = self._last_output_layer.output_shape
                
                if isinstance(layer_object, (nn_layers.Web)):
                    self._last_output_layer = layer_object

            if isinstance(layer_object, nn_layers.BatchNorm):
                self._last_output_layer.feeds_into_norm = True

            if isinstance(self.layers[-1], nn_layers.Web):
                self.layers[-1].output_layer = layer_object

        if isinstance(layer_object, nn_layers.Loss):
            if layer_object.loss_func == node_funcs.BCE:

                if self._last_output_layer.output_shape != 1:
                    raise Exception("Should use output layer of size 1 when using binary cross entropy loss,\
                                    decrease layer size to 1 or use CE (regular cross entropy)")

            if layer_object.loss_func == node_funcs.CE:
                if self._last_output_layer.output_shape < 2:
                    raise Exception("Should use cross entropy loss for multi-class classification, increase layer-size or use BCE")

            self.loss_layer = layer_object
        
        if isinstance(layer_object, nn_layers.Dropout):
            self._has_dropout = True

        self.layers.append(layer_object)

    def _val_structure(self):
        """validate the structure of the the NN"""
        if len(self.layers) == 0:
            raise Exception("No layers in network")

        if self.loss_layer is None:
            raise Exception("Please add a loss function")

    # Initialization

    def initialize_params(self):

        if not self.loss_layer:
            raise Exception("Neural net not complete, add loss layer before initializing params")
        
        for layer in self.layers:
            if layer.learnable:
                layer.initialize_params()

    # EPOCH ROUTINE

    def epoch_routine(self, epoch):
        if self._has_dropout:
            self._set_epoch_dropout_masks(epoch=epoch)

    def _set_epoch_dropout_masks(self, epoch):

        for layer in self.layers:
            if isinstance(layer, nn_layers.Dropout):
                layer.set_epoch_dropout_mask(epoch)
    
    def batch_total_pass(self, X_train, y_train, learning_rate, reg_strength):
        """forward propagation, backward propagation and parameter updates for gradient descent"""

        # forward pass
        self._forward_prop(X_train)

        # perform back prop to obtain gradients and update
        self._back_prop(X_train, y_train, learning_rate, reg_strength)

    # FORWARD ROUTINE

    def _forward_prop(self, X_train):
        
        input = X_train
        for layer in self.layers:
            input = layer.advance(input, forward_prop_flag=True)

    # BACK ROUTINE

    def _back_prop(self, X_train, y_train, learning_rate, reg_strength):
        
        loss_layer = True
        for layer in reversed(self.layers):
            if loss_layer:
                input_grad_to_loss = layer.back_up(y_train)
                loss_layer = False
            else:
                if layer.learnable:
                    input_grad_to_loss = layer.back_up(input_grad_to_loss, learning_rate=learning_rate, reg_strength=reg_strength)
                else:
                    input_grad_to_loss = layer.back_up(input_grad_to_loss)

    # COST/LOSS

    def cost(self, X, y_true, reg_strength):
        """Calculate the average loss depending on the loss function
        
        Args:
            X (numpy array) : examples to predict on (num_examples x num_features)
            y (numpy array) : true labels (num_examples x 1)

        Returns:
            cost (numpy array) : average loss given predictions on X and truth y
        
        """
        # calculate activation values for each layer (includes predicted values)
        y_pred = self.predict_prob(X)

        cost = self.loss_layer.get_cost(y_pred, y_true)

        # L2 regularization loss with Frobenius norm
        if reg_strength != 0: 
            cost = cost + reg_strength * sum(np.sum(layer._weights ** 2) for layer in self.layers if isinstance(layer, nn_layers.Web))

        return cost
    
    # INFERENCE/EVALUATION

    def predict_prob(self, X):
        """Obtain output layer activations
        
        Args: 
            X (numpy array) : examples (num_examples x num_features)
        
        Returns:
            cur_a (numpy array) : probabilities of output layer
        """

        # set the data as "a0"
        input = X

        # go through the layers and save the activations
        for layer in self.layers:
            input = layer.advance(input, forward_prop_flag=False)

        return input

    def predict_labels(self, X):
        """Predict labels of given X examples
        
        Args:
            X (numpy array) : array of examples by row

        Returns:
            predictions (numpy array) : array of prediction for each example by row
        
        """
        # calculate activation values for each layer (includes predicted values)
        final_activations = self.predict_prob(X)

        if isinstance(self.loss_layer.loss_func, node_funcs.BCE):
            predictions = final_activations > .5
        else:
            predictions = np.argmax(final_activations, axis=1)

        return predictions

    def factory_accuracy(self, eval_generator):
        """Gives accuracy for a given generator
        
        Args:
            eval_generator (PreDataGenerator) : some data generator of examples and labels
        
        Returns:
            Accuracy (float)
        """
        
        eval_right_sum = 0
        eval_len_sum = 0

        for X_eval, y_eval in eval_generator.generate():
            y_pred = self.predict_labels(X_eval)

            if not isinstance(self.loss_layer.loss_func, node_funcs.BCE):
                y_eval = np.argmax(y_eval, axis=1)

            eval_right_sum += (y_pred == y_eval).sum()
            eval_len_sum += X_eval.shape[0]
        
        accuracy = eval_right_sum / eval_len_sum

        return accuracy
    
    def accuracy(self, X, y):
        """Return accuracy for given X examples and y labels
        
        Args:
            X (numpy.ndarray) : examples/features (num_examples, ...)
        
        Returns:
            y (numpy.ndarray) : ground truth labels (num_examples, ...)
        """

        y_pred = self.predict_labels(X)

        if not isinstance(self.loss_layer.loss_func, node_funcs.BCE):
            y = np.argmax(y, axis=1)

        accuracy = (y_pred == y).sum() / X.shape[0]

        return accuracy
    
    def save_model(self, path):
        joblib.dump(self, path)

    @ classmethod
    def load_model(cls, path):
        potential_model = joblib.load(path)
        if not isinstance(potential_model, cls):
            raise TypeError(f"Loaded model must be of type {cls}")
        return potential_model

class RecurrentNN:
    pass