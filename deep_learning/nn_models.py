# model structures to pass into model manager
# have their own forward prop, epoch, and back prop routines

import numpy as np
from deep_learning import nn_layers, node_funcs, utils
from deep_learning.node import Node
import copy
import joblib

# TODO

# make sure the cell dict directed graph has no cycle somehow adn enforce full connection


# Create Joint models that are not just linear, multiple inputs/outputs allowed from each layer
## Make loop functionality for recurrent networks (just evaluate last??)
## Make recurrent NN submodel check for RNN Web layers
### only if autoregression should require RNN webs

## Add in a GRU
## Add teacher forcing functionality
## Add encoding functionality for functions such as translation

## bidirectional RNN will be two regular RNN with a "cap" for the output


# COMPLETED
## Make so dense layers can be a feedforward neural net
## get rid of last_output_layer by pushing bias decision to the interface, give each layer input and output dim
## calc input grad flag set to False on the MonoModel Pieces (changed structure)
# Just unravel a models layers as nodes? makes for simpler algorithms

# REJECTED

class MonoModelPiece(Node): 
    """Plain feed-forward neural network that can be used as a submodel"""

    def __init__(self, loss_required_flag=True, str_id=None):
        super().__init__(str_id)
        self.layers = []
        self.loss_layer = None

        self._has_fit = False
        self._has_dropout = False
        self.has_input = False

        self.loss_required_flag = loss_required_flag
    
    # STRUCTURE

    def add_layer(self, layer_object):

        if self.loss_layer is not None:
            raise Exception("Cannot add another layer after Loss layer")

        if isinstance(layer_object, nn_layers.InputLayer):
            self.has_input = True

        if len(self.layers) > 0:
            if layer_object.input_shape is not None:
                if layer_object.input_shape != self.layers[-1].output_shape:
                    raise Exception("Input shape must equal the output shape of the last layer")
            else:
                layer_object.input_shape = self.layers[-1].output_shape
            
            if isinstance(layer_object, nn_layers.Web):
                if isinstance(self.layers[-1], nn_layers.InputLayer):
                    layer_object.calc_input_grad_flag = False

        if isinstance(layer_object, nn_layers.Loss):
            self.loss_layer = layer_object
        
        if isinstance(layer_object, nn_layers.Dropout):
            self._has_dropout = True

        self.layers.append(layer_object)

        # set the dimensions of the las
        self.input_shape = utils.dim_size(self.layers[0].input_shape)
        self.output_shape = utils.dim_size(self.layers[-1].output_shape)

    def _val_structure(self):
        """validate the structure of the the NN"""

        if len(self.layers) == 0:
            raise Exception("No layers in network")

        if self.loss_layer is None and self.loss_required_flag:
            raise Exception("Please add a loss function")

    # INITIALIZATION

    def initialize_params(self):

        if not self.loss_layer and self.loss_required_flag:
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
        self.forward_prop(X_train)

        # perform back prop to obtain gradients and update
        self.back_prop(X_train, y_train, learning_rate, reg_strength)

    # FORWARD ROUTINE

    def forward_prop(self, X_train):
        
        input = X_train
        for layer in self.layers:
            input = layer.advance(input, forward_prop_flag=True)

        return input
    # BACK ROUTINE

    def back_prop(self, X_train, y_train, learning_rate, reg_strength):
        
        for layer in reversed(self.layers):
            if isinstance(layer, nn_layers.Loss):
                input_grad_to_loss = layer.back_up(y_train)
            else:
                if layer.learnable:
                    input_grad_to_loss = layer.back_up(input_grad_to_loss, learning_rate=learning_rate, reg_strength=reg_strength)
                else:
                    input_grad_to_loss = layer.back_up(input_grad_to_loss)

        return input_grad_to_loss
    
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

    @classmethod
    def load_model(cls, path):
        potential_model = joblib.load(path)
        if not isinstance(potential_model, cls):
            raise TypeError(f"Loaded model must be of type {cls}")
        return potential_model

class MonoModel(MonoModelPiece):
    """Stand alone mono model"""

    def add_layer(self, layer_object):

        if len(self.layers) == 0:
            if not isinstance(layer_object, nn_layers.InputLayer):
                raise Exception("First layer must be an Input Layer")
            
        super().add_layer(layer_object)

    # no need to return last outputs of forward prop and back prop

    def forward_prop(self, X_train):

        input = X_train
        for layer in self.layers:
            input = layer.advance(input, forward_prop_flag=True)
    
    def back_prop(self, X_train, y_train, learning_rate, reg_strength):
        
        for layer in reversed(self.layers):
            if isinstance(layer, nn_layers.Loss):
                input_grad_to_loss = layer.back_up(y_train)
            else:
                if layer.learnable:
                    input_grad_to_loss = layer.back_up(input_grad_to_loss, learning_rate=learning_rate, reg_strength=reg_strength)
                else:
                    input_grad_to_loss = layer.back_up(input_grad_to_loss)

class JointedModel:

    """Model that can be represented as a digraph with multiple paths,
    does not require a loss. Recurrency represented thorugh loops of state inputs.

    Attributes:
        cell_dict (dict) : {node -> target_nodes} graph dictionary that represents a cell
    """

    def __init__(self, cell_dict):
        
        self._has_loss = False

        # unravel models into layers
        cell_graph_layers_dict, self._connected_models = self.unravel_graph_layers(cell_dict)
        print(cell_graph_layers_dict)
        self.cell_layers_dict = cell_graph_layers_dict

        self._has_fit = False
        self._has_dropout = False

    @property
    def cell_layers_dict(self):
        return self._cell_layers_dict
    
    @property
    def recurrent_layers_dict(self):
        return self._recurrent_layers_dict

    def dfs_joint_search_compilation_helper(self, start_node, queue, cell_dict):
        """recursive function for sfs_joint_search_compilation

        Args:
            start_node (Layer, Model) : node to start search on
            queue (list) : list of nodes in queue to search
            cell_dict (dic) : cell_graph
        
        code in ### is not part of base dfs algorithm
        """

        # if self._recurrent_flag:
        #     recurrent_e = Exception("Model is recurrent, must use recurrent Webs")
        #     if isinstance(start_node, nn_layers.Web):
        #         if type(start_node) != nn_layers.RNN_Web:
        #                 raise recurrent_e

        if type(start_node) in [nn_layers.StackedInputLayer, nn_layers.InputLayer]:
            if self.data_node is None:
                self.data_node = start_node
            else:
                raise Exception("Can't have more than one data entry point")

        # add node to nodes list
        if start_node not in self.layers:
            self.layers.add(start_node)
        
        # enforce only one loss per cell
        if isinstance(start_node, nn_layers.Loss):
            if self._has_loss:
                raise Exception("Loss already set")
            else:
                self._has_loss = True

        # find all edges that enter the start_node
        incoming_edge_list = [edge for edge in self.cell_edge_list if edge[1] == start_node]

        if len(incoming_edge_list) > 1: # or could just condition on being a joint layer
            
            ### must be a joint layer 
            if not isinstance(start_node, nn_layers.SumLayer):
                raise Exception("A layer with multiple incoming connections must be a joint layer")
            ###

            # filter searched edges to edges 
            filter_edge_list_searched = [edge for edge in self.cell_edge_search_history if edge[1] == start_node]

            # if all incoming edges searched and has further connections
            if len(incoming_edge_list) != len(filter_edge_list_searched): 
                return 1
        
        # handle neighbors
        neighbors = cell_dict.get(start_node, [])

        # if connected to state input, have to have the state input come first so the input is not transformed

        for neighbor in neighbors:

            ###
            if isinstance(start_node, nn_layers.Loss):
                raise Exception("Loss nodes cannot have a neighbor")

            if type(neighbor) in [nn_layers.InputLayer, nn_layers.StackedInputLayer]:
                raise Exception("Neighbor can't be a data input layer")
            
            #if neighbor doesn't have an input shape then add one
            if neighbor.input_shape is None:
                neighbor.input_shape = start_node.output_shape
            else:
                assert neighbor.input_shape == start_node.output_shape, f"{start_node} and {neighbor} dimensions do not match"

            self.cell_edge_search_history.append((start_node, neighbor))
            ###

            if type(neighbor) != nn_layers.StateInputLayer:
                self.dfs_joint_search_compilation_helper(neighbor, queue, cell_dict)

    def dfs_joint_search_compilation(self, graph_dict):
        """Search through a model graph starting with an initial_queue of nodes to search.
        When the search reaches a joint, if all incoming connections have been searched, 
        and the joint has unsearched edges, the joint node will be added to the FIFO queue for search. 
        Edges are added to a list as tuples in the order that they are searched 
        
        Args:
            graph_dict (dict) : {input_nodes : list_of_neighbor_nodes} graph dictionary
            initial_queue (list) : list of nodes in order of how they should be searched

        Returns:
            cell_edge_search_history (list) : list of tuples representing edges searched in order of encountering those edges
        """

        # initialize cell_edge_search_history
        self.cell_edge_search_history = []
        self.layers = set()
        self.data_node = None

        # reverse graph dict for checking incoming nodes
        self.cell_edge_list = utils.graph_dict_to_edge_list(graph_dict)

        queue = self.start_nodes[:]
        while queue:
            cur_node = queue.pop(0)
            self.dfs_joint_search_compilation_helper(cur_node, queue, graph_dict)

    @cell_layers_dict.setter
    def cell_layers_dict(self, graph_dict_layers):
        self._val_graph_dict_layers(graph_dict_layers)
        self.start_nodes = [node for node in graph_dict_layers.keys() if isinstance(node, nn_layers.InputLayer)]
        self._val_start_nodes()

        self.dfs_joint_search_compilation(graph_dict_layers)

        self._cell_layers_dict = graph_dict_layers

    @staticmethod
    def unravel_graph_layers(graph_dict):
        """Replace models to their layers for processing. 
        Keep track of connected models in graph dict
        
        Args:
            graph_dict : {input_node -> [output_nodes]} dict, nodes are layers or models

        Returns:
            graph_layers_dict : {input_node -> [output_nodes]}, nodes are layers
            connected_models : 
        """

        graph_layers_dict = {}
        connected_models = set()

        for start_node, target_nodes in graph_dict.items():

            layer_list = []
            for target_node in target_nodes:
                # target node model connected to input by first layer
                if isinstance(target_node, MonoModelPiece):
                    
                    layer_list.append(target_node.layers[0])

                    if target_node not in connected_models:
                        connected_models.add(target_node)
                else:
                    layer_list.append(target_node)

            # start node model connected to output by last layer
            if isinstance(start_node, MonoModelPiece):
                connected_models.add(start_node)

                last_layer = start_node.layers[-1]
                graph_layers_dict[last_layer] = layer_list
                                      
            else:
                graph_layers_dict[start_node] = layer_list

            # handle unraveling the models
            for model in connected_models:
                for start_layer, target_layer in zip(model.layers[:-1], model.layers[1:]):
                    graph_layers_dict[start_layer] = [target_layer]

        return graph_layers_dict, connected_models

    @staticmethod
    def _val_graph_dict_layers(graph_dict_layers):
        
        for layer, target_layers in graph_dict_layers.items():
            if type(layer) != nn_layers.Splitter and len(target_layers) > 1:
                raise Exception("Layer cannot have more than one output node if it is not a splitter")


    def _val_start_nodes(self):
        """Validate the start nodes of the graph. Make sure there are start nodes and that they have input layers"""
        # make sure there are starting nodes
        if len(self.start_nodes) == 0:
            raise Exception("Network graph must have at least one starting node")
        # make sure starting nodes have some input shape declared

        for node in self.start_nodes:
            if not isinstance(node, nn_layers.InputLayer):
                raise Exception("Must declare input shape for all starting nodes")

    # INITIALIZATION

    def initialize_params(self):
        for layer in self.layers:
            if layer.learnable:
                layer.initialize_params()

            if not self._has_dropout:
                if type(layer) == nn_layers.BatchNorm:
                    self._has_dropout = True

    # EPOCH ROUTINES

    def epoch_routine(self, epoch):
        if self._has_dropout:
            self._set_epoch_dropout_masks(epoch=epoch)

    def _set_epoch_dropout_masks(self, epoch):

        for layer in self.layers:
            if isinstance(layer, nn_layers.Dropout):
                layer.set_epoch_dropout_mask(epoch)

    # BATCH TRAINING

    def forward_prop(self, X_train):
        
        if type(self.data_node) == nn_layers.StackedInputLayer:
            for timestep_data in X_train:
                self.data_node.store_input(timestep_data)

        elif type(self.data_node) == nn_layers.InputLayer:
            input_val = self.data_node.advance(X_train, forward_prop_flag=True)
            # flow turned on in class internally when advance

        print(self.data_node)
        # flowing handles when to stop this loo
        while self.data_node.flowing:
            # while self.data_node.flowing
            for start, target in self.cell_edge_search_history:
                
                if type(self.data_node) == nn_layers.InputLayer:
                    self.data_node.flowing = False
                
                # handles start
                if type(start) in [nn_layers.StateInputLayer, nn_layers.StackedInputLayer, nn_layers.SumLayer, nn_layers.Splitter]:
                    input_val = start.discharge_cell_output()

                print(f"input val {input_val}")

                # advance or store the cell input
                if type(target) in [nn_layers.SumLayer, nn_layers.StateInputLayer, nn_layers.Splitter]:
                    # print("storing input")
                    target.store_cell_input(input_val)
                else:
                    input_val = target.advance(input_val)
                
                print(type(target))
                print(input_val)

        return input_val

    def back_prop(self, X_train, y_train, learning_rate, reg_strength):
        pass
        
        # for our first pass we ignore getting outpts
        for target, start in reversed(self.edge):
            pass

class RecurrentNN:
    """Simple recurrent many-to-many neural network

    Attributes:
        Tx (int) : length of sequence of inputs
        Ty (int) : length of sequence of outputs
        xs_model : model that maps from input to state precursor
        so_model : model that maps from state to output
        ss_model : model that maps from state to state precursor
        bo : bias to add before output activation
        bs : bias to add before state activation
        s : the model state
    """
   
    def __init__(self, Tx, Ty,
                 xs_model,
                 so_model,
                 ss_model, 
                 state_activation=node_funcs.TanH):

        self.Tx = Tx
        self.Ty = Ty

        self.xs_model = xs_model
        self.so_model = so_model
        self.ss_model = ss_model

        self._state_activation = state_activation

        self._has_fit = False

    # setters and getters

    @property
    def Tx(self):
        return self._Tx
    
    @Tx.setter
    def Tx(self, Tx_cand):
        utils.pos_int(Tx_cand, "Tx")
        if Tx_cand < 2:
            raise ValueError("Tx should be greater than 1")
        self._Tx = Tx_cand

    @property
    def Ty(self):
        return self._Ty
    
    @Ty.setter
    def Ty(self, Ty_cand):
        utils.pos_int(Ty_cand, "Ty")
        if Ty_cand != 1 and Ty_cand != self.Tx:
            raise ValueError("Ty must be equal to Tx or set to 1")

        self._Ty = Ty_cand

    @property
    def xs_model(self):
        return self._xs_model

    @xs_model.setter
    def xs_model(self, xs_model_cand):
        assert type(xs_model_cand) == MonoModelPiece, "RNN submodels must be of type RecurrentNNSubModel"

        if xs_model_cand.track_input_layer_flag:
            raise Exception("Should not track input layer for input models")
        
        self._xs_model = xs_model_cand

    @property
    def so_model(self):
        return self._so_model

    @so_model.setter
    def so_model(self, so_model_cand):
        assert type(so_model_cand) == MonoModelPiece, "RNN submodels must be of type RecurrentNNSubModel"

        if not so_model_cand.loss_required_flag:
            raise Exception("so_model should have a loss")
        
        if not so_model_cand.track_input_layer_flag:
            raise Exception("Should track input layer for so_model")

        assert self.xs_model.output_shape == so_model_cand.input_shape, "State dimensions must match (dimensions must match between xs model output and so model input)"

        self._so_model = so_model_cand

    @property
    def ss_model(self):
        return self._ss_model

    @ss_model.setter
    def ss_model(self, ss_model_cand):
        assert type(ss_model_cand) == MonoModelPiece, "RNN submodels must be of type RecurrentNNSubModel"

        if not ss_model_cand.track_input_layer_flag:
            raise Exception("Should track input layer for ss_model")
    
        assert self.xs_model.output_shape == ss_model_cand.input_shape, "State dimensions must match (dimensions must match between xs model output and ss model input)"

        self._ss_model = ss_model_cand

    # INITIALIZE PARAMS

    def initialize_params(self):

        self.xs_model.initialize_params()
        self.so_model.initialize_params()
        self.ss_model.initialize_params()

        self._state_dim = self.ss_model.input_shape

        self.bs = np.zeros(self._state_dim)
        self.bo = np.zeros(self.so_model.output_shape)

    def batch_total_pass(self, X_train, y_train, learning_rate, reg_strength):
        """forward propagation, backward propagation through time and parameter updates for gradient descent"""

        # forward pass
        self.forward_in_time(X_train, y_train, learning_rate, reg_strength)

        # perform back prop to obtain gradients and update
        self.back_in_time(X_train, y_train, learning_rate, reg_strength)

    def forward_in_time(self, X_train, y_train, learning_rate, reg_strength):
        
        # initialize the state to a zero vector
        state = np.zeros(self._state_dim)

        for t in range(self.Tx):
            tx_inputs = X_train[:,t,:]

            xs_output = self.xs_model.forward_prop(tx_inputs)
            ss_output = self.ss_model.forward_prop(state)

            state = self.state_activation.forward(xs_output + ss_output + self.bs)
            
            # output if needed
            if t >= self.Tx - self.Ty:
                ty_outputs = y_train[:,t-(self.Tx-self.Ty),:]
                self.so_model.forward_prop(state)
                #dh_dL = 


    def back_in_time(self):
        pass


    

        