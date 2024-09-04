import numpy as np
from deep_learning import no_resources, node_funcs, utils
from deep_learning.node import Node
import time


# TODO

# have to fix how these timesteps are kept track of (namely in SumLayer), excess forward prop flags
# make joint layer a parent class, joint layer functionalities
# discharge cell output forward prop flag, probably could differentiate on learnabl
# input shape idx for concat layer is kind of weird 

# COMPLETED
# make use_bias term so does not have to be automatically decided
# make initialization a parameter so output layer does not need to be tracked for automatic initialization
# Create an input layer so (calc_input_grad_flag) can be easily set
# make input layers for data and also dummy variables (like first state in an RNN)
# make a data lock class like the Songo Locks (see sum layer)

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

        self.timestep = 0

    # INITIALIZE PARAMS

    def initialize_params(self):
        
        input_size = utils.dim_size(self.input_shape)

        rng = np.random.default_rng(self._seed)
        self._weights = rng.normal(size=(self.input_shape, self.output_shape)) * np.sqrt(self.init_factor / input_size)

        if self.use_bias:
            self._bias = np.zeros(shape=self.output_shape)

        self._accumulated_weights_grad_to_loss = None
        self._accumulated_bias_grad_to_loss = None

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
            # adding with padding to accommodate variable sequence lengths
            if self._accumulated_weights_grad_to_loss is None:
                self._accumulated_weights_grad_to_loss = weights_grad_to_loss
            else:
                utils.adding_with_padding(self._accumulated_weights_grad_to_loss, weights_grad_to_loss)

            # self._accumulated_weights_grad_to_loss += weights_grad_to_loss # these operations are expensive

            # for BPTT, if are processing the first reached input for the layer, then can update with accumulated gradient
            if len(self._input_stack) == 0:
                # print(f"Updating web on {self.str_id}")
                self._update_param(self._weights, self._accumulated_weights_grad_to_loss, learning_rate)
                self._accumulated_weights_grad_to_loss = None

            if self.use_bias:
                bias_grad_to_loss = self._calc_bias_grads(output_grad_to_loss)
                if self._accumulated_bias_grad_to_loss is None:
                    self._accumulated_bias_grad_to_loss = bias_grad_to_loss
                else:
                    utils.adding_with_padding(self._accumulated_bias_grad_to_loss, bias_grad_to_loss)
                #self._accumulated_bias_grad_to_loss += bias_grad_to_loss

                if len(self._input_stack) == 0:
                    # print(f"Updating bias on {self.str_id}")
                    self._update_param(self._bias, self._accumulated_bias_grad_to_loss, learning_rate)
                    self._accumulated_bias_grad_to_loss = None

        return input_grad_to_loss

    # CALCULATE GRADIENTS
    
    def _calc_weights_grads(self, output_grad_to_loss, input_val, reg_strength):

        # print(f"output grad to loss: {output_grad_to_loss.shape}")
        # print(f"input val: {input_val.shape}")

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

class StateInputLayer(SameDimLayer):
    """Holds initial state for a given state and subsequent transformations for that state
    
    Can change output sizes based on number of cell batches
    """

    learnable = False

    def __init__(self, dim, str_id=None):
        super().__init__(dim, str_id)
        self.cell_input = np.zeros(dim)
        self.cell_output_grad = None

    def store_cell_input(self, cell_input_val):
        self.cell_input = cell_input_val

    def discharge_cell_output(self, forward_prop_flag=True):
        return self.cell_input

    def store_cell_output_grad(self, cell_output_grad):
        self.cell_output_grad = cell_output_grad

    def discharge_cell_input_grad(self):
        return self.cell_output_grad

class StackedInputLayer(SameDimLayer):
    """data input for temporal models"""

    learnable = False

    def __init__(self, dim, str_id=None):
        super().__init__(dim, str_id)
        self.data_input_stack = []
        self.cell_output_grad_stack = []
        self.cell_input_batch_sizes = []
        self.trend = None

    def store_cell_input(self, input_val, track_input_size=False):
        # validate
        if input_val.shape[-1] != self.dim:
            raise Exception(f"Data dimension mismatch between input and set dimension, data is {input_val.shape} but should be {self.dim}")

        # check for monotonicity either increasing or decreasing
        if len(self.data_input_stack) >= 1:
            if self.trend == 1:
                assert len(input_val) >= len(self.data_input_stack[-1])
            elif self.trend == 0:
                assert len(input_val) <= len(self.data_input_stack[-1])
            elif self.trend == None:
                if len(input_val) > len(self.data_input_stack[-1]):
                    self.trend = 1
                elif len(input_val) < len(self.data_input_stack[-1]):
                    self.trend = 0

        self.data_input_stack.append(input_val)

        if track_input_size:
            self.cell_input_batch_sizes.append(len(input_val))

    def load_data(self, data_array, track_input_size=False, load_backwards=False):
        """Load data into input stack so it can be fetched for later use 
        
        Args:
            data_array (numpy array) : data to load in in size (num_examples, timesteps, data_for_timestep) 
            track_input (bool) : whether or not to track length (batch size) of the data 
            load_backwards (bool, default=False) : whether or not to load in data where the last element in data array will be read first
        """

        # print(sum(len(data_array[:,t,:]) for t in range(len(data_array.oha_list[-1]))))

        if len(self.data_input_stack) == 0:
            
            num_timesteps = data_array.shape[1]

            for t in range(num_timesteps):
                
                if load_backwards:
                    t = num_timesteps - t - 1

                timestep_data = data_array[:,t,:]
                # could be selecting wrong at predict prob

                # print(f"timestep data {timestep_data}")

                self.store_cell_input(timestep_data, track_input_size=track_input_size)
        else:
            raise Exception("Clear data from the input stack")

    def discharge_cell_output(self, forward_prop_flag=True):
        
        if len(self.data_input_stack) == 1:
            self.trend = None

        return self.data_input_stack.pop(0)

    @property
    def flowing(self):
        return len(self.data_input_stack) != 0

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
            # if self.loss_func == node_funcs.MSE:
            #     self._input_stack.append(input_val)
            self._output_stack.append(output)

        return output
    
    def get_total_loss(self, y_pred, y_true):

        return self.loss_func.forward(y_pred, y_true)

    def get_cost(self, y_pred, y_true):

        cost = np.mean(self.get_total_loss(y_pred, y_true))

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

    def __init__(self, use_bias=True, output_flag=False, str_id=None):
        super().__init__(str_id=str_id)

        # input stack for each cell
        self.input = None
        self.cell_output_grad_stack = []
        
        # whether or not splitter leads to an output
        self.output_flag = output_flag

    # keep track of timestep for upgrading bias

    def discharge_cell_output(self, forward_prop_flag=True):
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

        self.cur_timestep = 0

    def store_cell_input(self, input_val):
        """Place an input in the input stack"""
        #print(f"Storing cell input")
        #print(f"len sum input val {len(input_val)}")

        self.cell_input_stack.append(input_val)

    def initialize_params(self):
        
        if self.use_bias:
            self._bias = np.zeros(self.input_shape)
            self._accumulated_bias_grad_to_loss = None

    def discharge_cell_output(self, forward_prop_flag=True):
        if self.cell_input_stack:
            self.set_cell_output(forward_prop_flag)
        return self.output

    def set_cell_output(self, forward_prop_flag):
        """Set output before releasing"""

        if len(self.cell_input_stack) == 0:
            raise Exception("No layer inputs to produce an output")

        if forward_prop_flag:
            self.cur_timestep += 1

        self.num_inputs = len(self.cell_input_stack)

        self.output = 0
        while self.cell_input_stack:
            self.output += self.cell_input_stack.pop()

        if self.use_bias:
            self.output += self._bias
    
    def discharge_cell_input_grad(self):
        return self.input_grad

    def set_input_grad(self, output_grad_to_loss):
        self.input_grad = output_grad_to_loss

    def back_up(self, output_grad_to_loss, learning_rate, reg_strength, update_params_flag=True):
        
        self.set_input_grad(output_grad_to_loss)
        self.cur_timestep -= 1

        #print(f"backing up {self.cur_timestep}")

        # update params
        if update_params_flag and self.use_bias:
                
            bias_grad_to_loss = self._calc_bias_grads(output_grad_to_loss)
            if self._accumulated_bias_grad_to_loss is None:
                self._accumulated_bias_grad_to_loss = bias_grad_to_loss
            else:
                utils.adding_with_padding(self._accumulated_bias_grad_to_loss, bias_grad_to_loss)
            #self._accumulated_bias_grad_to_loss += bias_grad_to_loss

            if self.cur_timestep == 0:
                # print(f"updating bias on {self.str_id}")
                self._update_param(self._bias, self._accumulated_bias_grad_to_loss, learning_rate)
                self._accumulated_bias_grad_to_loss = None

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
        # if use_bias_cand:
        #     self.learnable = True
        # else:
        #     self.learnable = False

        self._use_bias = use_bias_cand

class ConcatLayer(Node):
    """Jointed Layer made for use in LSTM and variants"""
    
    learnable = False

    def __init__(self, output_shape, input_shapes=None, str_id=None):

        super().__init__(str_id=str_id)

        self.output_shape = output_shape
        self.input_shapes = input_shapes if input_shapes is not None else []
        self.input_shape_idx = 0

        self.cell_input_stack = []

        self.input_grads = []

        self.input_shapes_tracker = []
        self.input_shapes_tracker_temp = []
        
    def store_cell_input(self, input_val):
        
        if type(input_val) != np.ndarray:
            input_val = input_val.to_array()

        # validate input shapes appearing in right order
        correct_shape = self.input_shapes[self.input_shape_idx]
        self.input_shape_idx = (self.input_shape_idx + 1) % len(self.input_shapes)

        if input_val.shape[-1] != correct_shape:
            raise Exception(f"Input shape should be {correct_shape}, but {input_val.shape[-1]} was provided")
        
        # input shapes tracker
        self.input_shapes_tracker_temp.append(input_val.shape[-1])
        if len(self.input_shapes_tracker_temp) > len(self.input_shapes_tracker):
            self.input_shapes_tracker = self.input_shapes_tracker_temp[:]

        self.cell_input_stack.append(input_val)

    def discharge_cell_output(self, forward_prop_flag=True):

        output = np.hstack(self.cell_input_stack)
        # clear the output
        self.cell_input_stack = []
        self.input_shapes_tracker_temp = []

        # pad if needed (first timestep cell where no state input is provided)
        if output.shape[-1] < self.output_shape:
            output = utils.zero_lpad(output, self.output_shape - output.shape[-1])

        return output
    
    # backwards

    def store_cell_output_grad(self, output_grad_to_loss):
        if len(self.input_grads) > 0:
            raise Exception("Input grads already set")
        
        hsplit_idxs = np.cumsum(self.input_shapes_tracker[:-1])
        self.input_grads = np.hsplit(output_grad_to_loss, hsplit_idxs) # splits

    def discharge_cell_input_grad(self):
        # discharge in reverse order of input through splitting with right order of dimensions
        return self.input_grads.pop()
    
def ConcatLayerStack(self):
    pass


class MultLayer(SameDimLayer):
    """Piecewise multiplication layer often used for gates"""
    
    learnable = False

    def __init__(self, str_id=None):

        super().__init__(str_id=str_id)

        # list of inputs for the cell
        self.cell_input_stack = []
        # input to output grads for each factor
        self.cell_io_grads_list = []
        self.cur_cell_io_grads = None
        # output grad to loss for the current cel
        self.output_grad = None
        # cell product for the current cell
        self.cell_product = None

    def store_cell_input(self, input_val):
        self.cell_input_stack.append(input_val)

    def discharge_cell_output(self, forward_prop_flag=True):
        
        if len(self.cell_input_stack) > 1:
            product = utils.array_list_product(self.cell_input_stack)
        else:
            product = np.zeros(self.cell_input_stack[0].shape)

        self.save_io_grads(product)

        return product
    
    def save_io_grads(self, array_product):
        """calculate the io grads for the cell by input"""
        io_grads = []
        while self.cell_input_stack:
            # popping here puts io_grads in order of last input received to first input received for the cell
            input_val = self.cell_input_stack.pop()
            io_grad = array_product / input_val
            io_grads.append(io_grad)
        self.cell_io_grads_list.append(io_grads)

    def store_cell_output_grad(self, output_grad_val):
        # set up node for calculating cell input grads by input
        self.cur_cell_io_grads = self.cell_io_grads_list.pop()
        # set output grad val for the cell
        self.output_grad = output_grad_val

    def discharge_cell_input_grad(self):
        
        # fetch the correct io grad for the input last input received -> first input received
        io_grad = self.cur_cell_io_grads.pop(0)

        return self.output_grad * io_grad

class ComplementLayer(SameDimLayer):
    """1 - the input"""
    learnable = False

    def __init__(self, str_id=None):

        super().__init__(str_id=str_id)

    def advance(self, input_val, forward_prop_flag=True):
        return 1 - input_val
    
    def back_up(self, output_grad_to_loss):
        return -output_grad_to_loss
        