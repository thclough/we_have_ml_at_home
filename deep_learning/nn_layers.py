import numpy as np
from deep_learning import no_resources, node_funcs, utils
from deep_learning.node import Node
import time


# TODO


# make a data lock class like the Songo Locks
# make joint layer a parent class, joint layer functionalities


# COMPLETED
# make use_bias term so does not have to be automatically decided
# make initialization a parameter so output layer does not need to be tracked for automatic initialization
# Create an input layer so (calc_input_grad_flag) can be easily set
# make input layers for data and also dummy variables (like first state in an RNN)

# REJECTED
# Create an input layer so (ignore_input_grad) can then be forgotten (weird to calculate gradients this way)

# WEB LAYER
class Web(Node):
    """Weights layer to linearly transform inputs
    
    Attributes:
        init_factor (int) : numerator over input size to set standard deviation of normal distribution for sampling initial weight values
    
    """

    learnable = True
    
    def __init__(self, output_shape, input_shape=None, init_factor=1, use_bias=True, seed=100, str_id=None):
        
        self.calc_input_grad_flag = True
        # init factor two for ReLU
        self.init_factor = init_factor
        self.use_bias = use_bias

        super().__init__(str_id)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self._seed = seed

        self._weights = None
        self._bias = None

        self._input_stack = []

    # INITIALIZE PARAMS

    def initialize_params(self):
        
        input_size = utils.dim_size(self.input_shape)

        rng = np.random.default_rng(self._seed)
        self._weights = rng.normal(size=(self.input_shape, self.output_shape)) * np.sqrt(self.init_factor / input_size)

        if self.use_bias:
            self._bias = np.zeros(shape=self.output_shape)

        self._accumulated_weights_grad_to_loss = 0
        self._accumulated_bias_grad_to_loss = 0

    # PROPAGATE
    
    def advance(self, input_val, forward_prop_flag=True):
        """Move forward in the Neural net"""

        if forward_prop_flag:
            self._input_stack.append(input_val)

        if self.use_bias:
            output = input_val @ self._weights + self._bias
        else:
            output = input_val @ self._weights 

        return output

    def back_up(self, output_grad_to_loss, learning_rate, reg_strength, update_params_flag=True):
        
        if self.calc_input_grad_flag:
            input_grad_to_loss = output_grad_to_loss @ self._weights.T
        else:
            input_grad_to_loss = None

        # fetch the input
        input_val = self._input_stack.pop()

        # update params
        if update_params_flag and input_val is not None:
            
            weights_grad_to_loss = self._calc_weights_grads(output_grad_to_loss, input_val, reg_strength=reg_strength)
            self._accumulated_weights_grad_to_loss += weights_grad_to_loss # these operations are expensive

            # for BPTT, if are processing the first reached input for the layer, then can update with accumulated gradient
            if len(self._input_stack) == 0:
                self._update_param(self._weights, self._accumulated_weights_grad_to_loss, learning_rate)
                self._accumulated_weights_grad_to_loss = 0

            if self.use_bias:
                bias_grad_to_loss = self._calc_bias_grads(output_grad_to_loss)
                self._accumulated_bias_grad_to_loss += bias_grad_to_loss

                if len(self._input_stack) == 0:
                    self._update_param(self._bias, self._accumulated_bias_grad_to_loss, learning_rate)
                    self._accumulated_bias_grad_to_loss = 0

        return input_grad_to_loss

    # CALCULATE GRADIENTS
    
    def _calc_weights_grads(self, output_grad_to_loss, input_val, reg_strength):

        m = len(input_val)

        if reg_strength != 0:
            weights_grad_to_loss = input_val.T @ (output_grad_to_loss / m) + 2 * reg_strength * self._weights
        else:
            weights_grad_to_loss = input_val.T @ (output_grad_to_loss / m)
        
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

class RNN_Web(Web):
    """Specialized learnable linear transformation for RNN. Enables back propagation through time."""

    def __init__(self, output_shape, input_shape=None, init_factor=1, seed=100, use_bias=True, str_id=None):
        super().__init__(output_shape, input_shape, init_factor, seed, use_bias, str_id)

        # input stack in case replicated in BPTT
        # the web could be replicated many times in the model and needs to store all of its past seen inputs
        # LIFO stack
        self._input_stack = []
    
    def initialize_params(self):
        super().initialize_params()
        # for BPTT
        self._accumulated_weights_grad_to_loss = 0
        self._accumulated_bias_grad_to_loss = 0

    def advance(self, input, forward_prop_flag=True):
        """Move forward in the Neural net"""

        if forward_prop_flag:
            self._input_stack.append(input)

        if self.use_bias:
            output = input @ self._weights + self._bias
        else:
            output = input @ self._weights 

        return output

    def back_up(self, output_grad_to_loss, learning_rate, reg_strength, update_params_flag=True):
        
        if self.calc_input_grad_flag:
            input_grad_to_loss = output_grad_to_loss @ self._weights.T
        else:
            input_grad_to_loss = None

        # fetch the input
        input = self._input_stack.pop()

        # update params
        if update_params_flag and input is not None:
            
            weights_grad_to_loss = self._calc_weights_grads(output_grad_to_loss, input, reg_strength=reg_strength)
            self._accumulated_weights_grad_to_loss += weights_grad_to_loss # these operations are expensive

            # for BPTT, if are processing the first reached input for the layer, then can update with accumulated gradient
            if len(self._input_stack) == 0:
                self._update_param(self._weights, self._accumulated_weights_grad_to_loss, learning_rate)
                self._accumulated_weights_grad_to_loss = 0

            if self.use_bias:
                bias_grad_to_loss = self._calc_bias_grads(output_grad_to_loss)
                self._accumulated_bias_grad_to_loss += bias_grad_to_loss

                if len(self._input_stack) == 0:
                    self._update_param(self._bias, self._accumulated_bias_grad_to_loss, learning_rate)
                    self._accumulated_bias_grad_to_loss = 0

        return input_grad_to_loss

class SameDimLayer(Node):
    """Parent class for layers that do not change the dimensions of their processed data"""
    
    def __init__(self, dim=None, str_id=None):
        super().__init__(str_id)
        self.dim = dim

    @property
    def input_shape(self):
        return self._dim
    
    @input_shape.setter
    def input_shape(self, input_shape_cand):
        self._dim = input_shape_cand

    @property
    def output_shape(self):
        return self._dim
    
    @output_shape.setter
    def output_shape(self, output_shape_cand):
        self._dim = output_shape_cand

    @property
    def dim(self):
        return self._dim
    
    @dim.setter
    def dim(self, cand_dim):
        self._dim=cand_dim

class Activation(SameDimLayer):

    learnable = False

    def __init__(self, activation_func, str_id=None):
        
        super().__init__(dim=None, str_id=str_id)
        
        self.activation_func = activation_func

        self._input_stack = []

    def advance(self, input_val, forward_prop_flag=True):

        if forward_prop_flag:
            self._input_stack.append(input_val)

        return self.activation_func.forward(input_val)
    
    def back_up(self, output_grad_to_loss):

        input_val = self._input_stack.pop()

        input_grad_to_output = self.activation_func.backward(input_val)

        input_grad_to_loss = output_grad_to_loss * input_grad_to_output

        return input_grad_to_loss

# INPUT LAYERS

class InputLayer(SameDimLayer):
    """Base class for data input, parent class for more complex"""
    learnable = False

    def __init__(self, dim, str_id=None):
        super().__init__(dim, str_id=str_id)
        self.flowing=True

    def advance(self, input_val, forward_prop_flag=True):
        self.flowing = True
        return input_val
    
    def back_up(self, output_grad_to_loss):
        return output_grad_to_loss

class StateInputLayer(InputLayer):
    """Holds initial state for a given state and subsequent transformations for that state"""

    def __init__(self, dim, str_id=None):
        super().__init__(dim, str_id)
        self.cell_input = np.zeros(dim)
        self.cell_output_grad = None

    def store_cell_input(self, cell_input_val):
        self.cell_input = cell_input_val

    def discharge_cell_output(self):
        return self.cell_input

    def store_cell_output_grad(self, cell_output_grad):
        self.cell_output_grad = cell_output_grad

    def discharge_cell_input_grad(self):
        return self.cell_output_grad

class StackedInputLayer(InputLayer):
    """data input for temporal models"""

    def __init__(self, dim, str_id=None):
        super().__init__(dim, str_id)
        self.cell_input_stack = []
        self.cell_output_grad_stack = []

    def store_cell_input(self, input_val):
        if input_val.shape != self.dim:
            raise Exception("Data dimension mismatch between input and set dimension")
        self.cell_input_stack.append(input_val)

    def discharge_cell_output(self):
        return self.input_stack.pop(0)

    @property
    def flowing(self):
        return len(self.cell_input_stack) == 0

# LOSS LAYER

class Loss(SameDimLayer):
    
    learnable = False

    def __init__(self, activation_func, loss_func, str_id=None):
        
        self.activation_func = activation_func
        self.loss_func = loss_func

        super().__init__(str_id=str_id)

        self._input_stack = []
        self._output_stack = []
        
    def advance(self, input_val, forward_prop_flag=True):
        
        output = self.activation_func.forward(input_val)

        if forward_prop_flag:
            #self._input_stack.append(input)
            self._output_stack.append(output)

        return output
    
    def get_total_loss(self, y_pred, y_true):

        return self.loss_func.forward(y_pred, y_true)

    def get_cost(self, y_pred, y_true):

        cost = np.mean(self.get_total_loss(y_pred, y_true))

        # L2 regularization

        return cost

    def back_up(self, y_true):

        output = self._output_stack.pop()

        input_grad_to_loss = self.loss_func.backward(output, y_true)

        return input_grad_to_loss
    
    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, dim_cand):
        if dim_cand is not None:
            if self.loss_func == node_funcs.BCE:
                if dim_cand != 1:
                    raise Exception("Should use output layer of size 1 when using binary cross entropy loss,\
                                     decrease layer size to 1 or use CE (regular cross entropy)")
            
            if self.loss_func == node_funcs.CE:
                if dim_cand < 2:
                    raise Exception("Should use cross entropy loss for multi-class classification, increase layer-size or use BCE")
        
        self._dim = dim_cand

# EFFICIENCY LAYERS

class Dropout(SameDimLayer):
    
    learnable = False

    def __init__(self, keep_prob, seed=100, str_id=None):
        
        super().__init__(str_id=str_id)

        # validate the keep prob
        self._val_keep_prob(keep_prob)

        self._keep_prob = keep_prob
        self._seed = seed

        self._epoch_dropout_mask = None

    @staticmethod
    def _val_keep_prob(keep_prob):
        if keep_prob <=0 or keep_prob >= 1:
            raise ValueError("keep prob must be between 0 and 1 exclusive")

    def set_epoch_dropout_mask(self, epoch):
        """Create masks for the epoch so masks can stay consistent throughout an epoch, 
        but also differ from epoch to epoch"""

        mask_rng = np.random.default_rng(self._seed+epoch)

        self._epoch_dropout_mask = (mask_rng.random(self.input_shape) < self._keep_prob).astype(int)

    def advance(self, input_val, forward_prop_flag=True):
        if forward_prop_flag:
            output = (input_val * self._epoch_dropout_mask) / self._keep_prob
            return output
        else:
            return input_val
        
    def back_up(self, output_grad_to_loss):
        
        input_grad_to_loss = (output_grad_to_loss * self._epoch_dropout_mask) / self._keep_prob

        return input_grad_to_loss
        
class BatchNorm(SameDimLayer):
    """For use in feedforward neural networks, for recurrent neural networks, see LayerNorm. 
    Increases Lipschitzness (smoothness) of the loss surface for faster training"""

    learnable = True

    def __init__(self, seed=199, str_id=None):

        super().__init__(str_id=str_id)

        self._seed = 100

        self._scale = None
        self._shift = None
        
        self._inf_mean = None
        self._inf_var = None

        self._z_hat = None
    
    def initialize_params(self):

        input_size = utils.dim_size(self.input_shape)

        # scale and shift
        rng = np.random.default_rng(self._seed)
        self._scale = rng.normal(input_size) * np.sqrt(1 / input_size)
        self._shift = np.zeros(shape=input_size)

        # params for inference normalization
        self._inf_mean = np.zeros(shape=input_size)
        self._inf_var = np.zeros(shape=input_size)

    def advance(self, input_val, forward_prop_flag=True):
        
        if forward_prop_flag:
            # find batch mean and var
            self._batch_mean = input_val.mean(axis=0)
            self._batch_var = input_val.var(axis=0)

            # exponential average update batch mean and var
            self._inf_mean = .9 * self._inf_mean + .1 * self._batch_mean
            self._inf_var = .9 * self._inf_var + .1 * self._batch_var

            z_hat = self._z_hat = (input_val - self._batch_mean) / np.sqrt(self._batch_var + 10e-8)

        else:
            z_hat = (input_val - self._inf_mean) / np.sqrt(self._inf_var + 10e-8)
            
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

# JOINTED ARCHITECTURE/GATES

class Splitter(SameDimLayer):
    
    learnable = False

    def __init__(self, use_bias=True, str_id=None):
        super().__init__(str_id=str_id)

        # input stack for each cell
        self.input = None
        self.cell_output_grad_stack = []
    # keep track of timestep for upgrading bias

    def discharge_cell_output(self):
        if self.input is not None:
            return self.input
        else:
            raise Exception("No cell output set")

    def store_cell_input(self, input_val):
        self.input = input_val

    def store_cell_output_grad(self, output_grad):
        """Place an output grad to loss in the output stack"""
        self.cell_output_grad_stack.append(output_grad)

    def discharge_cell_input_grad(self):
        
        # set the cell output grad stack if multiplied inputs
        if self.cell_output_grad_stack:
            self.set_input_grad()

        # output grad to loss is the input grad to loss
        return self.input_grad

    def set_input_grad(self):
        """Set input gradients before releasing to potentially multiple input pathways"""

        if len(self.cell_output_grad_stack) == 0:
            raise Exception("No layer output gradients to produce output gradients")
        
        self.timestep -= 1

        self.input_grad = 0
        while self.cell_output_grad_stack:
            self.input_grad += self.cell_output_grad_stack.pop()

class SumLayer(SameDimLayer): # SOME TYPE OF JOINT
    """Merge model outputs from multiple inputs and sum"""
    learnable = True
    def __init__(self, use_bias=True, str_id=None):
        
        super().__init__(str_id=str_id)
        self.use_bias = use_bias

        # input stack for each cell
        self.cell_input_stack = []
        self.input_grad = None

        self.timestep = 0

    def store_cell_input(self, input_val):
        """Place an input in the input stack"""
        self.cell_input_stack.append(input_val)

    def initialize_params(self):
        
        if self.use_bias:
            self._bias = np.zeros(self.input_shape)
            self._accumulated_bias_grad_to_loss = 0

    def discharge_cell_output(self):
        if self.cell_input_stack:
            self.set_cell_output()
        return self.output

    def set_cell_output(self):
        """Set output before releasing"""

        if len(self.cell_input_stack) == 0:
            raise Exception("No layer inputs to produce an output")

        self.timestep += 1

        self.num_inputs = len(self.cell_input_stack)

        self.output = 0
        while self.cell_input_stack:
            self.output += self.cell_input_stack.pop()

        if self.use_bias:
            self.output += self._bias
    
    def discharge_input_grad(self):
        return self.input_grad

    def set_input_grad(self, output_grad_to_loss):
        self.input_grad = output_grad_to_loss

    def back_up(self, output_grad_to_loss, learning_rate, reg_strength, update_params_flag=True):
        
        self.set_input_grad(output_grad_to_loss)
        self.timestep -= 1
        # update params
        if update_params_flag and self.use_bias:
                
            bias_grad_to_loss = self._calc_bias_grads(output_grad_to_loss)
            self._accumulated_bias_grad_to_loss += bias_grad_to_loss

            if self.timestep == 0:
                self._update_param(self._bias, self._accumulated_bias_grad_to_loss, learning_rate)
                self._accumulated_bias_grad_to_loss = 0

        # output grad to loss is the input grad to loss
        return self.input_grad
    
    def _calc_bias_grads(self, output_grad_to_loss):
        
        bias_grad_to_loss = output_grad_to_loss.mean(axis=0)

        return bias_grad_to_loss

    def _update_param(self, param, grad, learning_rate):

        if isinstance(grad, no_resources.RowSparseArray):
            (learning_rate * grad).subtract_from_update(param)
        else:
            param -= (learning_rate * grad)

    @property
    def use_bias(self):
        return self._use_bias
    
    @use_bias.setter
    def use_bias(self, use_bias_cand):
        if use_bias_cand:
            self.learnable = True
        else:
            self.learnable = False

        self._use_bias = use_bias_cand


    

class SumLayer2(SameDimLayer): # SOME TYPE OF JOINT
    """Merge model outputs and sum"""

    def __init__(self, use_bias=True):
        
        super().__init__()
        self.use_bias = use_bias

        self._input_stack = []

        # input stack for each cell
        self._cell_input_stack = []

    def store_cell_input(self, input):
        """Place an input in the input"""

        self._cell_input_stack.append(input)

    def initialize_params(self):
        
        if self.use_bias:
            self._accumulated_bias_grad_to_loss = 0
            self._bias = np.zeros(self.input_shape)

    def advance(self, forward_prop_flag=False):
        
        output = 0
        for input in self._cell_input_stack:
            output += input

        if self.use_bias:
            output += self._bias

        if forward_prop_flag:
            self._input_stack.append(self._cell_input_stack)

        return output

    def back_up(self, output_grad_to_loss, learning_rate, reg_strength, update_params_flag=True):
        
        input_grad_to_loss = output_grad_to_loss @ self._weights.T

        # fetch the input
        input = self._input_stack.pop()

        # update params
        if update_params_flag and input is not None:
                
            bias_grad_to_loss = self._calc_bias_grads(output_grad_to_loss)
            self._accumulated_bias_grad_to_loss += bias_grad_to_loss

            if len(self._input_stack) == 0:
                self._update_param(self._bias, self._accumulated_bias_grad_to_loss, learning_rate)
                self._accumulated_bias_grad_to_loss = 0

        return input_grad_to_loss

    @property
    def use_bias(self):
        return self._use_bias
    
    @use_bias.setter
    def use_bias(self, use_bias_cand):
        if use_bias_cand:
            self.learnable = True
        else:
            self.learnable = False

        return use_bias_cand


class GateLayer(SameDimLayer):

    pass