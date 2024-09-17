from deep_learning import nn_architecture, nn_layers, node_funcs, initializers

class PlainRNN(nn_architecture.JointedModel):
    """Plain RNN (not deep), must specify the state to output model
    
    for model layers:
        s is state
        x is input

    """

    def __init__(self, state_size, data_input_size, so_model=None, backward=False):
        """
        state_size (int) : number of values in state size array
        data_input_size (int): dimension of the input data
        so_model (nn_architecture.MonoModelPiece, default=None) : model mapping from internal state to the final output of the cell
        backwards (bool, default=False) : whether or not the model predicts backwards in time (direction of the model), useful for Bidirectional
        """
        # state to state model
        ss_model = nn_architecture.MonoModelPiece()
        ss_model.add_layer(nn_layers.StateInputLayer(state_size, str_id="ss_input"))
        ss_model.add_layer(nn_layers.Web(state_size, str_id="ss_web", use_bias=False))

        # input to state model
        xs_model = nn_architecture.MonoModelPiece()
        xs_model.add_layer(nn_layers.StackedInputLayer(data_input_size, str_id="xs_input"))
        xs_model.add_layer(nn_layers.Web(state_size, str_id="xs_web", use_bias=False))
        
        ss_xs_sumjoint = nn_layers.SumLayer(str_id="ss_xs_sum_joint")
        tanh_activation = nn_layers.Activation(node_funcs.TanH, str_id="tanh_joint_activation")

        splitter = nn_layers.Splitter(str_id="tanh splitter", output_flag=True)

        # put model together in a graph
        graph_dict = {ss_model: [ss_xs_sumjoint], xs_model: [ss_xs_sumjoint], ss_xs_sumjoint: [tanh_activation], tanh_activation: [splitter], splitter: [ss_model]}

        super().__init__(graph_dict, output_structure=so_model, backward=backward)

class PlainLSTM(nn_architecture.JointedModel):
    """Plain LSTM (not deep between layers), must specify the input output model"""

    def __init__(self, state_size, data_input_size, io_model=None, concat=True, backward=False, rec_init=initializers.Orthogonal, data_init=initializers.GlorotUniform):
        """
        Args:
            state_size (int) : number of values in state size array
            data_input_size (int): dimension of the input data
            io_model (nn_architecture.MonoModelPiece or Node, default=None) : model mapping from internal state h to the final output of the cell
            concat (bool, default=True) : whether or not to concat input data with state size 
                (make false if you have a special data type for input data that will not work with numpy array concat)
            backwards (bool, default=False) : whether or not the 
        """

        forget_mask = nn_layers.MultLayer(str_id="forget mask")
        input_mask = nn_layers.MultLayer(str_id="input mask")
        output_mask = nn_layers.MultLayer(str_id="output mask")

        cell_sum = nn_layers.SumLayer(use_bias=False, str_id="cell sum")

        cell_splitter =  nn_layers.Splitter(str_id="cell state splitter")

        cell_activation = nn_layers.Activation(node_funcs.TanH, str_id="final cell state activation")

        cell_state_store = nn_layers.StateInputLayer(state_size, str_id="cell state store")

        state_output_splitter = nn_layers.Splitter(str_id="state output splitter", output_flag=True)

        internal_state_store = nn_layers.StateInputLayer(state_size, str_id="internal state store")

        data_input_layer = nn_layers.StackedInputLayer(data_input_size, str_id="data_input_lstm")

        if concat:
            
            concat_layer = nn_layers.ConcatLayer(data_input_size + state_size, str_id="concat_layer")
            input_splitter = nn_layers.Splitter(str_id="gate_splitter")

            forget_model = nn_architecture.MonoModelPiece()
            forget_model.add_layer(nn_layers.Web(state_size, input_shape=state_size + data_input_size, str_id="forget web"))
            forget_model.add_layer(nn_layers.Activation(node_funcs.Sigmoid, str_id ="forget sig"))

            input_model = nn_architecture.MonoModelPiece()
            input_model.add_layer(nn_layers.Web(state_size, input_shape=state_size + data_input_size, str_id="input web"))
            input_model.add_layer(nn_layers.Activation(node_funcs.Sigmoid, str_id ="input sig"))

            cand_model = nn_architecture.MonoModelPiece()
            cand_model.add_layer(nn_layers.Web(state_size, input_shape=state_size + data_input_size, str_id="cand web"))
            cand_model.add_layer(nn_layers.Activation(node_funcs.TanH, str_id ="cand tanh"))

            output_model = nn_architecture.MonoModelPiece()
            output_model.add_layer(nn_layers.Web(state_size, input_shape=state_size + data_input_size, str_id="output web"))
            output_model.add_layer(nn_layers.Activation(node_funcs.Sigmoid, str_id ="output sig"))

            graph_dict = {data_input_layer: [concat_layer], internal_state_store: [concat_layer], concat_layer:[input_splitter], input_splitter: [forget_model, input_model, cand_model, output_model],
                        forget_model: [forget_mask], cell_state_store: [forget_mask], input_model: [input_mask], cand_model: [input_mask], output_model: [output_mask], 
                        forget_mask: [cell_sum], input_mask: [cell_sum],
                        cell_sum: [cell_splitter], cell_splitter: [cell_activation, cell_state_store],
                        cell_activation: [output_mask],
                        output_mask: [state_output_splitter], state_output_splitter : [internal_state_store],
                        # connect to the next cell
                        internal_state_store : [concat_layer],
                        cell_state_store : [forget_mask]}
        
        else:
            # internal state input
            internal_state_input_splitter = nn_layers.Splitter(str_id="internal_state_input_splitter")

            forget_web = nn_layers.Web(state_size, input_shape=state_size, use_bias=False, str_id="forget web", initializer=rec_init)
            input_web = nn_layers.Web(state_size, input_shape=state_size, use_bias=False, str_id="input web", initializer=rec_init)
            cand_web = nn_layers.Web(state_size, input_shape=state_size, use_bias=False, str_id="cand web", initializer=rec_init)
            output_web = nn_layers.Web(state_size, input_shape=state_size, use_bias=False, str_id="output web", initializer=rec_init)

            # data input
            data_input_splitter = nn_layers.Splitter(str_id="data_input_splitter")

            forget_web_data = nn_layers.Web(state_size, input_shape=data_input_size, use_bias=False, str_id="forget web data", initializer=data_init)
            input_web_data = nn_layers.Web(state_size, input_shape=data_input_size, use_bias=False, str_id="input web data", initializer=data_init)
            cand_web_data = nn_layers.Web(state_size, input_shape=data_input_size, use_bias=False, str_id="cand web data", initializer=data_init)
            output_web_data = nn_layers.Web(state_size, input_shape=data_input_size, use_bias=False, str_id="output web data", initializer=data_init)
            # sum layers

            forget_sum = nn_layers.SumLayer(str_id="forget sum")
            input_sum = nn_layers.SumLayer(str_id="input sum")
            output_sum = nn_layers.SumLayer(str_id="output sum")
            cand_sum = nn_layers.SumLayer(str_id="cand sum")

            forget_act = nn_layers.Activation(node_funcs.Sigmoid, str_id="forget gate act")
            input_act = nn_layers.Activation(node_funcs.Sigmoid, str_id="input gate act")
            cand_act = nn_layers.Activation(node_funcs.TanH, str_id="cand act")
            output_act = nn_layers.Activation(node_funcs.Sigmoid, str_id="output gate act")

            graph_dict = {data_input_layer: [data_input_splitter], data_input_splitter : [forget_web_data, input_web_data, cand_web_data, output_web_data],
                          internal_state_store: [internal_state_input_splitter], internal_state_input_splitter : [forget_web, input_web, cand_web, output_web],
                          forget_web_data : [forget_sum], forget_web : [forget_sum], forget_sum : [forget_act], 
                          input_web_data : [input_sum], input_web : [input_sum], input_sum : [input_act],
                          cand_web_data : [cand_sum], cand_web : [cand_sum], cand_sum : [cand_act],
                          output_web_data : [output_sum], output_web : [output_sum], output_sum : [output_act],
                          forget_act: [forget_mask], cell_state_store: [forget_mask], input_act: [input_mask], cand_act: [input_mask], output_act: [output_mask], 
                          forget_mask: [cell_sum], input_mask: [cell_sum],
                          cell_sum: [cell_splitter], cell_splitter: [cell_activation, cell_state_store],
                          cell_activation: [output_mask],
                          output_mask: [state_output_splitter], state_output_splitter:[internal_state_store],
                          # connect to the next cell
                          cell_state_store : [forget_mask]}

        super().__init__(graph_dict, output_structure=io_model, backward=backward)

class PlainGRU(nn_architecture.JointedModel):
    """Plain Gated Recurrent Unit (no deep networks between layers), must specify input->output model"""

    def __init__(self, state_size, data_input_size, io_model=None, concat=True, backward=False):
        """
        Args:
            state_size (int) : number of values in state size array
            data_input_size (int): dimension of the input data
            io_model (nn_architecture.MonoModelPiece or Node, default=None) : model mapping from internal state h to the final output of the cell
            concat (bool, default=True) : whether or not to concat input data with state size 
                (make false if you have a special data type for input data that will not work with numpy array concat)
            backwards (bool, default=False) : whether or not the 
        """
        # common elements between concat layer and non-concat layer

        data_input_layer = nn_layers.StackedInputLayer(data_input_size, str_id="data_input_gru")
        data_input_splitter = nn_layers.Splitter(data_input_size, str_id="data_splitter_gru")
        state_input_splitter = nn_layers.Splitter(str_id="state_input_splitter")

        update_gate_splitter = nn_layers.Splitter(str_id="update_gate_splitter")

        # into getting can values

        cand_gate_mask = nn_layers.MultLayer(str_id = "cand_gate_mask")

        # calculating new ht 

        subtract_layer = nn_layers.ComplementLayer(str_id= "subtract")

        maintain_mask = nn_layers.MultLayer(str_id = "maintain_mask")
        update_mask = nn_layers.MultLayer(str_id = "update_mask")

        final_sum = nn_layers.SumLayer(str_id="final_sum")

        state_output_splitter = nn_layers.Splitter(str_id="internal_state_splitter", output_flag=True)

        internal_state_store = nn_layers.StateInputLayer(state_size, str_id="internal state store")

        if concat:
            
            concat_layer = nn_layers.ConcatLayer(data_input_size + state_size, str_id="concat_layer_gru")

            update_cand_splitter = nn_layers.Splitter(str_id="update_cand_splitter")

            update_gate_model = nn_architecture.MonoModelPiece()
            update_gate_model.add_layer(nn_layers.Web(state_size, input_shape=state_size + data_input_size, str_id="update_gate_web"))
            update_gate_model.add_layer(nn_layers.Activation(node_funcs.Sigmoid, str_id="update_gate_act"))

            cand_gate_model = nn_architecture.MonoModelPiece()
            cand_gate_model.add_layer(nn_layers.Web(state_size, input_shape=state_size + data_input_size, str_id="cand_gate_web"))
            cand_gate_model.add_layer(nn_layers.Activation(node_funcs.Sigmoid, str_id="cand_gate_act"))

            cand_concat_layer = nn_layers.ConcatLayer(data_input_size + state_size, str_id="cand_concat_layer")

            cand_model = nn_architecture.MonoModelPiece()
            cand_model.add_layer(nn_layers.Web(state_size, input_shape=state_size + data_input_size, str_id="cand_model_web", use_bias=False))
            cand_model.add_layer(nn_layers.Activation(node_funcs.TanH, str_id="cand_model_act"))

            graph_dict = {data_input_layer : [data_input_splitter], data_input_splitter : [concat_layer, cand_concat_layer], internal_state_store: [state_input_splitter],
                        state_input_splitter : [concat_layer, cand_gate_mask, maintain_mask], concat_layer : [update_cand_splitter],
                        update_cand_splitter: [update_gate_model, cand_gate_model], update_gate_model: [update_gate_splitter], cand_gate_model : [cand_gate_mask], cand_gate_mask:[cand_concat_layer],
                        cand_concat_layer : [cand_model], cand_model : [update_mask],  update_gate_splitter : [subtract_layer, update_mask], subtract_layer : [maintain_mask],
                        maintain_mask: [final_sum], update_mask:[final_sum], final_sum : [state_output_splitter], state_output_splitter: [internal_state_store]}

        else: # no concat
            
            # update gate model into two webs sum and activate
            data_update_gate_web = nn_layers.Web(state_size, input_shape=data_input_size, use_bias=False, str_id="data_update_gate_web")
            state_update_gate_web = nn_layers.Web(state_size, input_shape=state_size, use_bias=False, str_id="state_update_gate_web")
            update_gate_sum = nn_layers.SumLayer(str_id="update_gate_sum")
            update_gate_activation = nn_layers.Activation(node_funcs.Sigmoid, str_id="update_gate_activation")
            
            # cand gate model into two webs sum and activate
            data_cand_gate_web = nn_layers.Web(state_size, input_shape=data_input_size, use_bias=False, str_id="data_cand_gate_web")
            state_cand_gate_web = nn_layers.Web(state_size, input_shape=state_size, use_bias=False, str_id="state_cand_gate_web")
            cand_gate_sum = nn_layers.SumLayer(str_id="cand_gate_sum")
            cand_gate_activation = nn_layers.Activation(node_funcs.Sigmoid, str_id="cand_gate_activation")

            # cand model broken into two webs sum and activate
            data_cand_web = nn_layers.Web(state_size, input_shape=data_input_size, use_bias=False, str_id="data_cand_web")
            state_cand_web = nn_layers.Web(state_size, input_shape=state_size, use_bias=False, str_id="state_cand_web")
            cand_sum = nn_layers.SumLayer(str_id="cand_sum")
            cand_activation = nn_layers.Activation(node_funcs.TanH, str_id="cand_activation")

            graph_dict = {data_input_layer : [data_input_splitter], data_input_splitter : [data_update_gate_web, data_cand_gate_web, data_cand_web],
                          internal_state_store : [state_input_splitter], state_input_splitter: [state_update_gate_web, state_cand_gate_web, cand_gate_mask, maintain_mask],
                          # update gate models
                          data_update_gate_web : [update_gate_sum], data_cand_gate_web : [update_gate_sum], update_gate_sum : [update_gate_activation],
                          # cand gate model
                          data_cand_gate_web : [cand_gate_sum], state_cand_gate_web : [cand_gate_sum], cand_gate_sum : [cand_gate_activation], 
                          # cand model
                          cand_gate_activation : [cand_gate_mask], cand_gate_mask : [state_cand_web],
                          data_cand_web : [cand_sum], state_cand_web: [cand_sum], cand_sum : [cand_activation],
                          # elements for final sum
                          update_gate_activation : [update_gate_splitter], update_gate_splitter: [subtract_layer, update_mask], 
                          cand_activation : [update_mask], subtract_layer : [maintain_mask],
                          # final sum
                          update_mask : [final_sum], maintain_mask : [final_sum],
                          # output
                          final_sum : [state_output_splitter], state_output_splitter:[internal_state_store]
                          }

        super().__init__(graph_dict, output_structure=io_model, backward=backward)
