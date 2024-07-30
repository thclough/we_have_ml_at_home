import numpy as np
from deep_learning import no_resources
from deep_learning import node_funcs
from deep_learning import utils

import time

# WEB LAYER
class Web:
    """Weights layer to linearly transform activation values"""

    learnable = True
    
    def __init__(self, output_shape, input_shape=None, seed=100):
        
        self.input_layer_flag = False
        self.output_layer = None
        self.feeds_into_norm = False

        self.input_shape = input_shape
        self.output_shape = output_shape

        self._seed = seed

        self._weights = None
        self._bias = None

        self._input = None

    # INITIALIZE PARAMS

    def initialize_params(self):

        input_size = utils.dim_size(self.input_shape)

        if isinstance(self.output_layer, (Activation, Loss)):
            if isinstance(self.output_layer.activation_func, (node_funcs.ReLU, node_funcs.LeakyReLU)): 
                factor = 2
            else:
                factor = 1
        else:
            factor = 1

        rng = np.random.default_rng(self._seed)
        self._weights = rng.normal(size=(self.input_shape, self.output_shape)) * np.sqrt(factor / input_size)
        if not self.feeds_into_norm:
            self._bias = np.zeros(shape=self.output_shape) 
    
    # PROPAGATE

    def advance(self, input, forward_prop_flag=True):
        """Move forward in the Neural net"""

        if forward_prop_flag: # don't save the input if the input layer
            self.input = input

        if self.feeds_into_norm:
            output = input @ self._weights 
        else: 
            output = input @ self._weights + self._bias

        return output

    def back_up(self, output_grad_to_loss, learning_rate, reg_strength, update_params_flag=True):
        
        if not self.input_layer_flag:
            input_grad_to_loss = output_grad_to_loss @ self._weights.T
        else:
            input_grad_to_loss = None

        # update params
        if update_params_flag and self.input is not None:
            
            weights_grad_to_loss = self._calc_weights_grads(output_grad_to_loss, reg_strength=reg_strength)
            self._update_param(self._weights, weights_grad_to_loss, learning_rate)

            if not self.feeds_into_norm:
                bias_grad_to_loss = self._calc_bias_grads(output_grad_to_loss)
                self._update_param(self._bias, bias_grad_to_loss, learning_rate)

        # discharge the input
        self.input = None

        return input_grad_to_loss
    
    # CALCULATE GRADIENTS
    
    def _calc_weights_grads(self, output_grad_to_loss, reg_strength):

        m = len(self.input)

        if reg_strength != 0:
            weights_grad_to_loss = self.input.T @ (output_grad_to_loss / m) + 2 * reg_strength * self._weights
        else:
            weights_grad_to_loss = self.input.T @ (output_grad_to_loss / m)
        
        return weights_grad_to_loss
    
    def _calc_bias_grads(self, output_grad_to_loss):
        
        bias_grad_to_loss = output_grad_to_loss.mean(axis=0)

        return bias_grad_to_loss

    # UPDATE PARAMS

    def _update_param(self, param, grad, learning_rate):

        if isinstance(grad, no_resources.RowSparseArray):
            (learning_rate * grad).subtract_from_update(param)
        else:
            param -= (learning_rate * grad)
        
class Activation:

    learnable = False

    def __init__(self, activation_func):
        
        self.activation_func = activation_func

        self.input = None

    def advance(self, input, forward_prop_flag=True):

        if forward_prop_flag:
            self.input = input

        return self.activation_func.forward(input)
    
    def back_up(self, output_grad_to_loss):

        input_grad_to_output = self.activation_func.backward(self.input)

        input_grad_to_loss = output_grad_to_loss * input_grad_to_output

        return input_grad_to_loss
    
# LOSS LAYER

class Loss:
    
    def __init__(self, activation_func, loss_func):
        
        self.activation_func = activation_func
        self.loss_func = loss_func

        self.input = None
        self.output = None

        self.learnable = False
        
    def advance(self, input, forward_prop_flag=True):
        
        output = self.activation_func.forward(input)
        if forward_prop_flag:
            self.input = input 
            self.output = output

        return output
    
    def get_total_loss(self, y_pred, y_true):

        return self.loss_func.forward(y_pred, y_true)

    def get_cost(self, y_pred, y_true):

        cost = np.mean(self.get_total_loss(y_pred, y_true))

        # L2 regularization

        return cost

    def back_up(self, y_true):

        input_grad_to_loss = self.loss_func.backward(self.output, y_true)

        self.input = None
        self.output = None

        return input_grad_to_loss
    
# EFFICIENCY LAYERS

class Dropout:
    
    learnable = False

    def __init__(self, keep_prob, seed=100):
        
        # validate the keep prob
        self._val_keep_prob(keep_prob)

        self._keep_prob = keep_prob
        self._seed = seed

        self._epoch_dropout_mask = None

        self.input_shape = None

    @staticmethod
    def _val_keep_prob(keep_prob):
        if keep_prob <=0 or keep_prob >= 1:
            raise ValueError("keep prob must be between 0 and 1 exclusive")

    def set_epoch_dropout_mask(self, epoch):
        """Create masks for the epoch so masks can stay consistent throughout an epoch, 
        but also differ from epoch to epoch"""

        mask_rng = np.random.default_rng(self._seed+epoch)

        self._epoch_dropout_mask = (mask_rng.random(self.input_shape) < self._keep_prob).astype(int)

    def advance(self, input, forward_prop_flag=True):
        if forward_prop_flag:
            output = (input * self._epoch_dropout_mask) / self._keep_prob
            return output
        else:
            return input
        
    def back_up(self, output_grad_to_loss):
        
        input_grad_to_loss = (output_grad_to_loss * self._epoch_dropout_mask) / self._keep_prob

        return input_grad_to_loss
        
class BatchNorm:
    learnable = True

    def __init__(self, seed=199):
        self._seed = 100

        self._scale = None
        self._shift = None
        
        self._inf_mean = None
        self._inf_var = None

        self._z_hat = None

        self.input_shape = None
    
    def initialize_params(self):

        input_size = utils.dim_size(self.input_shape)

        # scale and shift
        rng = np.random.default_rng(self._seed)
        self._scale = rng.normal(input_size) * np.sqrt(1 / input_size)
        self._shift = np.zeros(shape=input_size)

        # params for inference normalization
        self._inf_mean = np.zeros(shape=input_size)
        self._inf_var = np.zeros(shape=input_size)

    def advance(self, input, forward_prop_flag=True):
        
        if forward_prop_flag:
            # find batch mean and var
            self._batch_mean = input.mean(axis=0)
            self._batch_var = input.var(axis=0)

            # exponential average update batch mean and var
            self._inf_mean = .9 * self._inf_mean + .1 * self._batch_mean
            self._inf_var = .9 * self._inf_var + .1 * self._batch_var

            z_hat = self._z_hat = (input - self._batch_mean) / np.sqrt(self._batch_var + 10e-8)

        else:
            z_hat = (input - self._inf_mean) / np.sqrt(self._inf_var + 10e-8)
            
        output = self._scale * z_hat + self._shift

        return output
    
    def back_up(self, output_grad_to_loss, learning_rate, reg_strength, update_params_flag=True):
        
        m = len(output_grad_to_loss)

        # find original input grad to loss
        dloss_dz_hat = output_grad_to_loss * self._scale
        input_grad_to_loss = (m * dloss_dz_hat - np.sum(dloss_dz_hat, axis=0) - self._z_hat * np.sum(dloss_dz_hat * self._z_hat, axis=0)) / (m * np.sqrt(self._batch_var + 10e-8)) 

        if update_params_flag:

            scale_grad_to_loss = self._calc_scale_grad(output_grad_to_loss)
            self._update_param(self._scale, scale_grad_to_loss, learning_rate)

            shift_grad_to_loss = self._calc_shift_grad(output_grad_to_loss)
            self._update_param(self._shift, shift_grad_to_loss, learning_rate)

        self._batch_mean = None
        self._batch_var = None
        self._z_hat = None

        return input_grad_to_loss

    def _calc_scale_grad(self, output_grad_to_loss):

        scale_grad_to_loss = np.mean(output_grad_to_loss * self._z_hat, axis = 0)

        return scale_grad_to_loss

    def _calc_shift_grad(self, output_grad_to_loss):
        
        shift_grad_to_loss = np.mean(output_grad_to_loss, axis=0)

        return shift_grad_to_loss
    
    def _update_param(self, param, grad, learning_rate):

        if isinstance(grad, no_resources.RowSparseArray):
            (learning_rate * grad).subtract_from_update(param)
        else:
            param -= (learning_rate * grad)





