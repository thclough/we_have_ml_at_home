# model structures to pass into model manager
# have their own forward prop, epoch, and back prop routines

import numpy as np
from deep_learning import nn_layers, node_funcs, utils, no_resources
from deep_learning.node import Node
import copy
import joblib

# TODO

# make sure the cell dict directed graph has no cycle somehow ang enforce full connection
# could implement ufunc subtraction from one hot array 
# could separate edge validation from the joint search compilation
# concat layer reset is kind of weird, checking for order of input shapes

# could edge select output model for if only want one output

## Make recurrent NN submodel check for RNN Web layers


## Add teacher forcing functionality
## Add encoding functionality for functions such as translation

## bidirectional RNN will be two regular RNN with a "cap" for the output


# COMPLETED
## Make so dense layers can be a feedforward neural net
## get rid of last_output_layer by pushing bias decision to the interface, give each layer input and output dim
## calc input grad flag set to False on the MonoModel Pieces (changed structure)
## Just unravel a models layers as nodes? makes for simpler algorithms
## Make loop functionality for recurrent networks (just evaluate last??)
## only if loop should require RNN webs used input laters
## # fix flow forward architecture
# Create Joint models that are not just linear, multiple inputs/outputs allowed from each layer
# allow for LSTM model to be created
## Add in a GRU

# REJECTED

class MonoModelPiece(Node): 
    """Plain feed-forward neural network that can be used as a submodel"""

    def __init__(self, loss_required_flag=True, str_id=None):
        super().__init__(str_id)
        self.layers = []
        self.loss_layer = None

        self.has_fit = False
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
    does not require a loss. Recurrency represented through loops of state inputs.

    Attributes:
        cell_dict (dict) : {node -> target_nodes} graph dictionary that represents a cell
        backward (bool) : whether model 
    """

    def __init__(self, cell_dict, backward=False, output_structure=None):
        
        self._has_loss = False
        self.assembled = False

        # unravel models into layers
        cell_graph_layers_dict, self.connected_models = self.unravel_graph_layers(cell_dict)

        #print(cell_graph_layers_dict)
        self.cell_layers_dict = cell_graph_layers_dict

        if output_structure is not None:
            self.add_output_structure(output_structure)

        self.has_fit = False
        self._has_dropout = False

        self.timestep = 0

        self.backwards = backward

        

    @property
    def has_fit(self):
        return self._has_fit
    
    @has_fit.setter
    def has_fit(self, has_fit_cand):
        for model in self.connected_models:
            model.has_fit = has_fit_cand
        
        self._has_fit = has_fit_cand

    @property
    def cell_layers_dict(self):
        return self._cell_layers_dict
    
    @property
    def recurrent_layers_dict(self):
        return self._recurrent_layers_dict

    # first graph set
    
    @cell_layers_dict.setter
    def cell_layers_dict(self, cell_layers_dict_cand):
        
        self._val_graph_dict_layers(cell_layers_dict_cand)

        possible_data_start_nodes = [node for node in cell_layers_dict_cand.keys() if type(node) in (nn_layers.InputLayer, nn_layers.StackedInputLayer)]
        self.state_start_nodes = [node for node in cell_layers_dict_cand.keys() if type(node) == nn_layers.StateInputLayer]
        possible_output_nodes = [node for node in cell_layers_dict_cand.keys() if type(node) == nn_layers.Splitter and node.output_flag == True]

        self._one_node(possible_data_start_nodes, "data start")
        self._one_node(possible_output_nodes, "output")

        self.data_node = possible_data_start_nodes[0]
        self.output_node = possible_output_nodes[0]

        self._cell_layers_dict = cell_layers_dict_cand

    # edit the graph
    def add_output_structure(self, structure):
        """Set the output layers stemming from the output splitter
        
        Args:
            structure (nn_models.MonoModelPiece or Node) : layer(s) or mod

        """
        if self.assembled:
            raise Exception("Cannot change computation graph because it is already set, set output structure before assembly")
        
        if type(structure) == MonoModelPiece:
            more_layers_dict, more_connected_models = self.unravel_graph_layers({self.output_node: [structure]})
        
            for model in more_connected_models:
                if model not in self.connected_models:
                    self.connected_models.add(model)
                else:
                    raise Exception("Model given already a connected model")

            for start_layer, end_layers in more_layers_dict.items():
                if start_layer == self.output_node:
                    self.cell_layers_dict[self.output_node] += end_layers
                else:
                    self.cell_layers_dict[start_layer] = end_layers
                
        elif type(structure) == Node:
            if structure not in self.cell_layers_dict[self.output_node]:
                self.cell_layers_dict[self.output_node].append(structure)
    # Final assembly

    def assemble(self):
        """finalize the computation graph"""
        
        state_to_data_paths, data_to_state = self.dfs_joint_search_compilation(self.cell_layers_dict)

        # separate state "alleys" from the data nodes, this does not seems to be really needed
        self.reg_cell_edge_order, state_alley_paths = self.separate_state_alleys(data_to_state)
    
        # connect the paths
        cell_connection_paths = utils.join_paths(state_alley_paths, state_to_data_paths)

        # concat the paths
        self.concat_connection_edge_order = []
        for path in cell_connection_paths:
            self.concat_connection_edge_order += path

        # set dimensions of concat layer
        if any(type(layer) == nn_layers.ConcatLayer for layer in self.layers):
            self.set_concat_input_dims()

        self.assembled = True

    @staticmethod
    def unravel_graph_layers(graph_dict):
        """Replace models by their layers for processing. 
        Keep track of connected models in graph dict
        
        Args:
            graph_dict : {input_node -> [output_nodes]} dict, nodes are layers or models

        Returns:
            graph_layers_dict : {input_node -> [output_nodes]}, nodes are layers
            connected_models : models in the graph
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
                raise Exception(f"Layer {layer} cannot have more than one output node if it is not a splitter")

    def _one_node(self, cand_nodes, node_cat_name):
        """Validate the nodes of the graph. Make sure there is exactly one"""
        # make sure there are starting nodes
        if len(cand_nodes) == 0 or len(cand_nodes) > 1:
            raise Exception(f"Network graph must have exactly one {node_cat_name} node")

    def dfs_joint_search_compilation(self, graph_dict):
        """Search through a model graph starting with an initial_queue of nodes to search.
        When the search reaches a joint, if all incoming connections have been searched, 
        and the joint has unsearched edges, the joint node will be added to the FIFO queue for search. 
        Edges are added to a list as tuples in the order that they are searched.
        Different types of graphs are created with different start nodes:
            state_start_edge_order : paths that start with state inputs and end with the first splitter
            data_edge_order : graph that starts with data input node and ends with terminal nodes
        
        Args:
            graph_dict (dict) : {input_nodes : list_of_neighbor_nodes} graph dictionary

        """

        # initialize cell_edge_search_history
        cell_edge_search_history = []
        self.layers = set()

        state_to_data_paths = []
        data_to_state = []

        # reverse graph dict for checking incoming nodes
        self.cell_edge_list = utils.graph_dict_to_edge_list(graph_dict)

        # search the state start nodes first and gather their branches
        for cur_node in self.state_start_nodes:
            state_to_data_path = []
            self.dfs_joint_search_compilation_helper(cur_node, graph_dict, cell_edge_search_history, state_to_data_path)
            state_to_data_paths.append(state_to_data_path)

        self.dfs_joint_search_compilation_helper(self.data_node, graph_dict, cell_edge_search_history, data_to_state)

        # validate connectivity()
        if len(self.cell_edge_list) != len(cell_edge_search_history):
            raise Exception("Could not traverse entire graph, graph may not be connected or all input nodes may not have been provided")

        return state_to_data_paths, data_to_state
    
    def dfs_joint_search_compilation_helper(self, start_node, cell_dict, search_history, store_edge_list):
        """recursive function for sfs_joint_search_compilation

        Args:
            start_node (Layer, Model) : node to start search on
            cell_dict (dict) : cell_graph
            store_edge_list (list) : list of edges to add traversed edges to
        
        code in ### is not part of base dfs algorithm
        """
            
        # add node to nodes list
        if start_node not in self.layers:
            self.layers.add(start_node)

        if isinstance(start_node, nn_layers.Loss):
            if self._has_loss:
                raise Exception("Can only have one loss")
            else:
                self._has_loss = True
                self.loss_layer = start_node

        # find all edges that enter the start_node
        incoming_edge_list = [edge for edge in self.cell_edge_list if edge[1] == start_node]

        # if have reached an intersection, time to decide if to move on, or more incoming paths need to be searched
        if len(incoming_edge_list) > 1:
            # print(f"had multiple incoming: {start_node}")
            ### must be a joint layer 
            if not hasattr(start_node, "cell_input_stack"):
                raise Exception("A layer with multiple incoming connections must have a cell_input_stack")
            ###

            # filter searched edges to edges coming into the edge of interest
            filter_edge_list_searched = [edge for edge in search_history if edge[1] == start_node]

            # print(f"search history: {search_history}")
            # print(f"incoming: {filter_edge_list_searched}")

            # if all incoming edges have not been searched yet
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
            if type(neighbor) != nn_layers.ConcatLayer:
                if neighbor.input_shape is None:
                    neighbor.input_shape = start_node.output_shape
                else:
                    assert neighbor.input_shape == start_node.output_shape, f"{start_node} and {neighbor} dimensions do not match"

            #self.cell_edge_search_history.append((start_node, neighbor))
            search_history.append((start_node, neighbor))
            store_edge_list.append((start_node, neighbor))
            ###

            # do not want to leak into next cell before current cell is traversed
            if type(neighbor) != nn_layers.StateInputLayer:
                self.dfs_joint_search_compilation_helper(neighbor, cell_dict, search_history, store_edge_list)

    def set_concat_input_dims(self):
        """Special function to store input dims of concat layer in order"""
        for start, target in self.reg_cell_edge_order + self.concat_connection_edge_order:
            if type(target) == nn_layers.ConcatLayer:
                target.input_shapes.append(start.output_shape)
                # print(f"{target} shape: {target.input_shapes}")

    def separate_state_alleys(self, ordered_edges):
        """Separate the the regular cell edges from the paths that lead to state inputs
        
        Args:
            ordered_edges (list of 2-tuples) : list of edges to separate
        
        Returns:
            regular_cell_edge_order (list of 2-tuples) : data input cell without connections to next state
            state_alley_paths (list of list of 2-tuples) : splitter to state input paths (transitions to a new cell) concatenated
        """

        # initialize
        regular_cell_edge_order_reversed = []
        state_alley_paths = []

        on_alley = False
        # start searching reversed 
        for start, target in reversed(ordered_edges):
            
            if type(target) == nn_layers.StateInputLayer:
                on_alley = True
                state_alley_path_reversed = []
            if on_alley:
                state_alley_path_reversed.append((start, target))
                if type(start) == nn_layers.Splitter:
                    on_alley = False
                    state_alley_paths.append(state_alley_path_reversed[::-1])
            else:
                regular_cell_edge_order_reversed.append((start,target))

        regular_cell_edge_order = regular_cell_edge_order_reversed[::-1]

        return regular_cell_edge_order, state_alley_paths
    
    # INITIALIZATION

    def _val_structure(self):
        """For nn driver"""
        pass

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

    def batch_total_pass(self, X_train, y_train, learning_rate, reg_strength):
        if not self.assembled:
            raise Exception("Have not assembled the final graph yet, use assemble to set the graph")
        
        # forward pass
        self.forward_prop(X_train)

        # perform back prop to obtain gradients and update
        self.back_prop(X_train, y_train, learning_rate, reg_strength)

    def forward_prop(self, X_train):
        self.flow_forward_helper(X_train, forward_prop_flag=True)

    def flow_forward_helper(self, X, forward_prop_flag=False, output_hold=None, generation_hold=None):
        #self.flow_forward(X_train, True)
        if type(self.data_node) == nn_layers.StackedInputLayer:
            self.data_node.load_data(X, track_input_size=forward_prop_flag, load_backwards=self.backwards)
            input_val = None
        elif type(self.data_node) == nn_layers.InputLayer:
            input_val = self.data_node.advance(X)
            # flow turned on in class internally when advance

        # reset concat layers for proper validation
        self.reset_concat_layers()

        # launch the forward prop
        while self.data_node.flowing:
            if forward_prop_flag:
                self.timestep += 1

            # set the edge order
            if len(self.data_node.data_input_stack) > 1:
                edge_order = self.reg_cell_edge_order + self.concat_connection_edge_order
            else:
                edge_order = self.reg_cell_edge_order 
        
            input_val = self.flow_forward(edge_order, input_val=input_val, forward_prop_flag=forward_prop_flag, output_hold=output_hold, generation_hold=generation_hold)

    def flow_forward(self, edge_order, input_val=None, forward_prop_flag=False, output_hold=None, generation_hold=None):
        """General process for push data through the Jointed Model
        
        Args:
        """ 
        # decide on the correct edge order

        for start, target in edge_order:
            # print(f"course: {start}, {target}")

            # turn off data flow if just one entry
            if type(self.data_node) == nn_layers.InputLayer:
                self.data_node.flowing = False
            
            # handles start
            if type(start) in (nn_layers.StackedInputLayer, nn_layers.SumLayer, nn_layers.Splitter, nn_layers.ConcatLayer, nn_layers.MultLayer):
                input_val = start.discharge_cell_output(forward_prop_flag)
            elif type(start) == nn_layers.StateInputLayer: # for variable length sequences
                if generation_hold is None:
                    # peek at the next batch size
                    output_dim = len(self.data_node.data_input_stack[0])
                else:
                    output_dim = len(generation_hold[-1])

                # limit or expand state to match next cell batch size, will call this accordion
                input_val = utils.accordion(start.discharge_cell_output(forward_prop_flag), output_dim)

            # print(f"input val shape {input_val.shape}")

            # advance or store the cell input
            if type(target) in (nn_layers.SumLayer, nn_layers.Splitter, nn_layers.ConcatLayer, nn_layers.MultLayer):
                # print("storing input")
                target.store_cell_input(input_val)
            elif type(target) == nn_layers.StateInputLayer:
                if self.data_node.flowing:
                    target.store_cell_input(input_val)
            else:
                if self._has_loss:
                    if target == self.loss_layer:
                        input_val = target.advance(input_val, forward_prop_flag)

                        if output_hold is not None:
                            output_hold.append(input_val)

                        # store and add to the data stack if data generation
                        if not self.data_node.flowing and generation_hold is not None:
                            #print(f"input_val {input_val}")
                            if self.loss_layer.loss_func == node_funcs.MSE:
                                sampled = input_val
                            elif self.loss_layer.loss_func == node_funcs.BCE:
                                sampled = [[np.random.binomial(1, p=dist)] for dist in input_val]
                                if len(sampled) > 1:
                                    sampled = np.vstack(sampled)
                            else:
                                sampled = [[np.random.choice(input_val.shape[-1], p=dist)] for dist in input_val]
                                if len(sampled) > 1:
                                    sampled = np.vstack(sampled)
                                # convert to oha if needed
                                if type(generation_hold[-1]) == no_resources.OneHotArray:
                                    #print(sampled)
                                    sampled = no_resources.OneHotArray(shape=(len(sampled),input_val.shape[-1]), idx_array=sampled)
                            generation_hold.append(sampled)
                    else:
                        input_val = target.advance(input_val, forward_prop_flag)
                else:    
                    input_val = target.advance(input_val, forward_prop_flag)

    def back_prop(self, X_train, y_train, learning_rate, reg_strength):
        # loss reached flag to indicate when to start calculating output gradients and updating layers
        last_cell_flag = True
        
        while self.timestep > 0:
            if last_cell_flag:
                edge_order = self.reg_cell_edge_order
                last_cell_flag = False
                cell_batch_size = self.data_node.cell_input_batch_sizes.pop()
            else:
                edge_order = self.reg_cell_edge_order + self.concat_connection_edge_order

                cell_batch_size = self.data_node.cell_input_batch_sizes.pop() if self.data_node.cell_input_batch_sizes else None
            
            self.timestep -= 1

            # handling variable length sequence
            
            for target, start in reversed(edge_order):
                # print(f"back course : {start}, {target}")
                # HANDLE START
                if isinstance(start, nn_layers.Loss):
                    # get correct dimension
                    # fix first y train
                    t_idx = -self.timestep - 1 if self.backwards else self.timestep
                    if isinstance(y_train, (np.ndarray, no_resources.OneHotTensor)):

                        y_train_t = y_train[:,t_idx,:] # :output length on first if dealing with numpy array with variable sequences
                        
                        if type(y_train_t) != np.ndarray:
                            y_train_t = y_train_t.to_array()

                    else: # to deal with uneven sequences on list like forms
                        y_train_t = np.array([y_train_m[t_idx] for y_train_m in y_train if len(y_train_m) > self.timestep])

                    input_grad_to_loss = start.back_up(y_train_t)
                
                if type(start) in (nn_layers.Splitter, nn_layers.SumLayer, nn_layers.ConcatLayer, nn_layers.MultLayer):
                    input_grad_to_loss = start.discharge_cell_input_grad()

                elif type(start) == nn_layers.StateInputLayer:
                    # variable lengths sequences
                    input_grad_to_loss = start.discharge_cell_input_grad()
                    after_length = len(input_grad_to_loss)

                    input_grad_to_loss = utils.accordion(input_grad_to_loss, cell_batch_size)

                # HANDLE TARGET

                if type(target) in (nn_layers.Splitter, nn_layers.StateInputLayer, nn_layers.ConcatLayer, nn_layers.MultLayer):
                    target.store_cell_output_grad(input_grad_to_loss)
                elif type(target) != nn_layers.StackedInputLayer:
                    if target.learnable:
                        input_grad_to_loss = target.back_up(input_grad_to_loss, learning_rate=learning_rate, reg_strength=reg_strength)
                    else:
                        input_grad_to_loss = target.back_up(input_grad_to_loss)

        return input_grad_to_loss

    def reset_concat_layers(self):
        """reset the concat layers for proper validation"""
         # reset concat input dims
        for layer in self.layers:
            if type(layer) == nn_layers.ConcatLayer:
                layer.input_shape_idx = 0
                layer.input_grads = []

    # INFERENCE/EVALUATION

    def generation(self, X, cutoff_length):
        """Generate outputs from a single example
        
        Args:
            X (numpy.ndarray) : input example(s) to generate from, in (num_examples, time_periods, output_dims)
            cutoff_length (int) : max length of output sequence

        Returns:
            output_sequence (list) : generated sequence of the model
        """

        if len(X.shape) != 3:
            raise Exception("X must be in shape (num_examples, time_periods, output_dims) form")
        
        generation_sequence = []
        for t in range(X.shape[1]):
            generation_sequence.append(X[:,t,:])

        # reset concat layers for proper validation
        self.reset_concat_layers()

        # append first input before taken out
        self.data_node.load_data(X, load_backwards=self.backwards)

        while len(generation_sequence) < cutoff_length:

            if not self.data_node.flowing:
                self.data_node.store_cell_input(generation_sequence[-1])

            # set the edge order
            if len(self.data_node.data_input_stack) > 1 or len(generation_sequence) < cutoff_length - 1:
                edge_order = self.reg_cell_edge_order + self.concat_connection_edge_order
            else:
                edge_order = self.reg_cell_edge_order 

            self.flow_forward(edge_order=edge_order, generation_hold=generation_sequence)

        return generation_sequence

    def predict_prob(self, X):
        """Accumulates output
        and shifts (timesteps, num_examples, categories) to (num_examples, timesteps, categories)"""
        
        output_hold = []
        self.flow_forward_helper(X, forward_prop_flag=False, output_hold=output_hold)

        # assumed sorted from largest to smallest
        if len(output_hold[0]) != len(output_hold[-1]): # uneven sequence lengths
            
            output_hold_final = self.uneven_swap_axes(output_hold)

                #output_hold_final.append([output[m] for output in output_hold if m < len(output)])
        else: # same sequence lengths, represent numpy array
            output_hold = np.array(output_hold)
            output_hold_final = output_hold.swapaxes(0,1)

        return output_hold_final

    def uneven_swap_axes(self, output_hold):
        """Shifts (timesteps, num_examples, categories) output to (num_examples, timesteps, categories) output"""

        one_val_flag = len(output_hold[0][0]) == 1

        output_hold_final = []
        max_batch_size = max(len(output_hold[0]),len(output_hold[-1])) # for either increasing or decreasing lengths

        for m in range(max_batch_size):
            m_list = []
            for output in output_hold:
                if one_val_flag:
                    output_list = output.flatten()
                else:
                    output_list = output.tolist()
                if m < len(output):
                    m_list.append(output_list[m])

            output_hold_final.append(m_list)

        return output_hold_final
    
    def predict_labels(self, X):

        # calculate activation values for each layer (includes predicted values)
        final_activations = self.predict_prob(X)

        if isinstance(self.loss_layer.loss_func, node_funcs.BCE):
            predictions = final_activations > .5
        else:
            predictions = np.argmax(final_activations, axis=-1)

        return predictions
    
    def cost(self, X, y_true, reg_strength):
        """Calculate the average loss depending on the loss function
        
        Args:
            X (numpy array) : examples to predict on (num_examples x num_features)
            y_true (numpy array) : true labels (num_examples x 1)

        Returns:
            cost (numpy array) : average loss given predictions on X and truth y
        
        """
        # calculate activation values for each layer (includes predicted values)
        y_pred = self.predict_prob(X)

        # flatten y_pred and y_true to make compatible with past infrastructure
        y_pred = utils.flatten_batch_outputs(y_pred)

        if self.backwards:
            y_true = utils.flip_time(y_true)

        y_true = utils.flatten_batch_outputs(y_true)

        # print(f"len y_pred {len(y_pred)}")
        # print(f"len y_true {len(y_true)}")

        cost = self.loss_layer.get_cost(y_pred, y_true)

        # L2 regularization loss with Frobenius norm
        if reg_strength != 0: 
            cost = cost + reg_strength * sum(np.sum(layer._weights ** 2) for layer in self.layers if isinstance(layer, nn_layers.Web))

        return cost
    
    def save_model(self, path):
        joblib.dump(self, path)

    @classmethod
    def load_model(cls, path):
        potential_model = joblib.load(path)
        if not isinstance(potential_model, cls):
            raise TypeError(f"Loaded model must be of type {cls}")
        return potential_model
    
class Bidirectional:
    
    def __init__(self, forward_model, backwards_model):
        """Bidirectional sequential model where each """

        self.forward_model = forward_model

        self.backward_model = backwards_model

        # check if models have the same input and output dims

        if not self.forward_model.data_node.input_shape ==  self.forward_model.data_node.input_shape:
            raise Exception("Forwards and backwards models must have the same data input dimension")
            

    # setter and getters
    @property
    def forward_model(self):
        return self._forward_model

    @forward_model.setter
    def forward_model(self, forward_model_cand):
        
        if forward_model_cand.has_loss:
            raise Exception("Forward model should not have a loss, only ")

        self._forward_model = forward_model_cand

    @property
    def backward_model(self):
        return self._backward_model

    @backward_model.setter
    def backward_model(self, backward_model_cand):

        if backward_model_cand.backward != True:
            raise Exception("Backward model should have backward set to True")
        
        if backward_model_cand.has_loss:
            raise Exception("Backward model should not have a loss")

        self._backward_model = backward_model_cand
