# handles data factories and model

import time
import numpy as np
import joblib
from deep_learning import node_funcs, learning_funcs, utils
import matplotlib.pyplot as plt
import deep_learning.data_factory_blueprints

# TODO

# register all submodels as having a fit

class NNDriver:
    """Model manager to handle training and evaluation of a model"""

    def __init__(self, model):
        self.model = model
        
        self._learning_rates = []
        self._reg_strengths = []
        self._batch_sizes = []

        self._stable_constant = 10e-8

    def fit(self,
            data_factory,
            learning_scheduler=learning_funcs.ConstantRate(1),
            reg_strength=0,
            num_epochs=30,
            epoch_gap=5,
            batch_prob=.01,
            batch_seed=100,
            model_path=None,
            display_path=None,
            verbose=True, 
            display=True):
        
        self._fit_clear(batch_prob, batch_seed, verbose)

        self.refine(data_factory=data_factory,
                    learning_scheduler=learning_scheduler,
                    reg_strength=reg_strength,
                    num_epochs=num_epochs,
                    epoch_gap=epoch_gap,
                    model_path=model_path,
                    display_path=display_path,
                    verbose=verbose, 
                    display=display)
    
    def _fit_clear(self, batch_prob, batch_seed, verbose):

        # validate inputs def validate_structure
        self.model._val_structure()
        
        # clear data for the first fit
        self.model.has_fit = True
        self._epoch = 0

        self._learning_scheduler = None

        self._train_costs = []
        self._dev_costs = []

        self._learning_rates = []
        self._reg_strengths = []
        self._batch_sizes = []

        self._batch_prob = batch_prob
        self._batch_seed = batch_seed

        self._rounds = []

        self._train_collection = None
        self._dev_collection = None

        # get the initial weights and biases
        if verbose:
            print("WARNING: Creating new set of params")

        self.model.initialize_params()
        
    def refine(self,
            data_factory,
            learning_scheduler=None,
            reg_strength=None,
            num_epochs=15,
            epoch_gap=5,
            model_path=None,
            display_path=None,
            verbose=True, 
            display=True):
        
        if not self.model.has_fit:
            raise Exception("Please fit the model before refining")

        if not data_factory.train_generator._train_generator:
            raise Exception("Given train chunk must be a valid train chunk (_train_generator attribute set to True)")

        self.data_factory = data_factory

        if learning_scheduler is not None:
            self._learning_scheduler = learning_scheduler
        # set chunks
        batch_size = data_factory.train_generator.batch_size

        self._dev_flag = False
        if data_factory.dev_generator is not None:
            self._dev_flag = True

        # add rounds
        self._rounds.append(self._epoch)

        # update attributes if needed
        if reg_strength is not None:
            self.reg_strength = reg_strength
        
        # initialize the display
        if display:
            fig, ax = self._initialize_epoch_plot()

        start_epoch = self._epoch
        end_epoch = self._epoch + num_epochs

        for epoch in range(start_epoch, end_epoch):
            print(f"Epoch: {epoch}")
            epoch_start_time = time.time()

            # update learning rate
            self.learning_rate = self._learning_scheduler.get_learning_rate(epoch)

            # update epoch data
            self._learning_rates.append(self.learning_rate)
            self._reg_strengths.append(self.reg_strength)
            self._batch_sizes.append(batch_size)

            # if has epoch routine
            self.model.epoch_routine(epoch=epoch)

            # set a seed for sampling batches for loss
            rng2 = np.random.default_rng(self._batch_seed)
            # start_gen = time.time()
            for X_train, y_train in self.data_factory.train_generator.generate():
                # end_gen = time.time()
                # print(f"gen time {end_gen-start_gen}")

                start_batch = time.time()
                self.model.batch_total_pass(X_train, y_train, self.learning_rate, self.reg_strength)
                end_batch = time.time()
                print(f"batch time: {end_batch-start_batch}")
                
                if verbose:
                    if rng2.binomial(1, self._batch_prob):
                        sampled_batch_loss = self.model.cost(X_train, y_train, self.reg_strength)

                        print(f"\t Sampled batch loss: {sampled_batch_loss}")
                
                #start_gen = time.time()

            epoch_end_time = time.time()
            
            if verbose:
                print(f"Epoch completion time: {(epoch_end_time-epoch_start_time) / 3600} Hours")
            
            # record costs after each epoch gap
            if epoch % epoch_gap == 0:
                gap_start_time = time.time()
                
                epoch_train_cost = self.generator_cost(self.data_factory.train_generator, self.reg_strength)
                self._train_costs.append(epoch_train_cost)

                if verbose:
                    
                    print(f"\t Training cost: {epoch_train_cost}")

                if self._dev_flag:
                    epoch_dev_cost = self.generator_cost(self.data_factory.dev_generator, self.reg_strength)
                    self._dev_costs.append(epoch_dev_cost)
                    if verbose:
                        print(f"\t Dev cost: {epoch_dev_cost}")
                else:
                    self._dev_costs.append(None)

                if display:
                    self._update_epoch_plot(fig, ax, epoch, end_epoch)

                gap_end_time = time.time()

                if verbose:
                    print(f"Gap completion time: {(gap_end_time-gap_start_time) / 3600} Hours")
            else:
                self._train_costs.append(None)
                self._dev_costs.append(None)

            self._epoch += 1
            if model_path:
                self.model.save_model(model_path)

            if display and display_path:
                fig.savefig(display_path)
        
        # if this doesn't happen, axis hold onto a "ghost" artist/collection if saved and reloaded, cannot add to figs in next refine
        ax.cla()
        plt.close()

    # PROP
    
    # EPOCH PLOT 
    def _initialize_epoch_plot(self):
        #plt.close()
        fig, ax = plt.subplots()
        ax.set_title(f"Average Loss ({self.model.loss_layer.loss_func.name}) vs. Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Loss")
        if len(self._rounds) > 1:
            for round, pos in enumerate(self._rounds):
                ax.axvline(x=pos, linestyle="--", alpha=.5, c="grey")
                ax.text(x=pos+.1, y=.75, 
                        color="grey",
                        s=f"Round {round}", 
                        rotation=90, 
                        verticalalignment='top', 
                        horizontalalignment='left',
                        transform=ax.get_xaxis_transform())
            
            # when adding to a new figure, the transforms/scale of the Path Collections 
            # somehow are in axes coordinates (0 to 1 across the axis), you have to change to data coordinates (true data of the axis)
            self._train_collection.set_offset_transform(ax.transData)
            ax.add_collection(self._train_collection)
            if self._dev_flag:
                self._dev_collection.set_offset_transform(ax.transData)
                ax.add_collection(self._dev_collection)

        return fig, ax

    def _update_epoch_plot(self, fig, ax, epoch, num_epochs):
        """Updates training plot to display average losses

        Args:
            fig (matplotlib.pyplot.figure) : figure containing plot axis
            ax (matplotlib.pyplot.axis) : axis that contains line plots
            epoch (int) : epoch number to graph new data for
            num_epochs (int) : total number of epochs
        """
        if not self._train_collection:
            self._train_collection = ax.scatter(range(0,epoch+1), self._train_costs, marker="x", c="red", alpha=.5, label="Average training loss")
            ax.legend()
        else:
            self._update_scatter(self._train_collection, range(0,epoch+1), self._train_costs)
        if self._dev_flag:
            if not self._dev_collection:
                self._dev_collection = ax.scatter(range(0,epoch+1), self._dev_costs, marker="x", c="blue", alpha=.5, label="Average dev loss")
                ax.legend()
            else:
                self._update_scatter(self._dev_collection, range(0,epoch+1), self._dev_costs)

        max_val = max([d for d in self._dev_costs if d is not None] + [t for t in self._train_costs if t is not None])
        ax.set(xlim=[-.5,num_epochs], ylim=[min(0, max_val*2),max(0, max_val*2)])

        plt.pause(.2)

    @staticmethod
    def _update_scatter(collection, new_x, new_y):
        offsets = np.c_[new_x, new_y]
        collection.set_offsets(offsets)

    # COST/LOSS

    def generator_cost(self, eval_generator, reg_strength):
        """"Calculate loss on the eval chunk"""

        if isinstance(eval_generator, deep_learning.data_factory_blueprints.MiniBatchAssembly):
            
            cost = self.model.cost(eval_generator.X, eval_generator.y, reg_strength)

        else:

            loss_sum = 0
            length = 0

            for X_data, y_data in eval_generator.generate():
                y_probs = self.model.predict_prob(X_data)

                # flatten if necessary
                y_probs = utils.flatten_batch_outputs(y_probs)
                y_data = utils.flatten_batch_outputs(y_data)

                generator_cost_sum = np.sum(self.model.loss_layer.get_total_loss(y_probs, y_data))
                chunk_length = y_data.shape[0]

                loss_sum += generator_cost_sum
                length += chunk_length
            
            cost = loss_sum / length

        return cost
    
    # PERFORMANCE
    
    def class_matrix(self, eval_generator):
        """Create classification matrix for the given chunk.
        
        Args:
            eval_generator (Chunk) : chunk to generate classification matrix for. 
                True labels along 0 axis and predicted labels along 1st axis.

        Returns:
            sorted_labels_key (dict) : {label value : idx} dictionary key for matrices
            report (numpy array) : classification matrix for the chunk
        """
        # hold classification matrix coordinates (true label, predicted label) -> count
        class_matrix_dict = {}

        # separate report dict and labels in case labels are not 0 indexes
        labels = set()

        for X_eval, y_eval in eval_generator.generate():
            
            self._modify_labels_and_class_matrix_dict(X_eval,y_eval,labels,class_matrix_dict)
        
        sorted_labels_key, class_matrix, f1s = self._create_report_items(class_matrix_dict, labels)

        return sorted_labels_key, class_matrix, f1s
    
    def class_report(self):
        """Generate classification report for all of the generators in the manager"""
        class_matrix_dict_of_dicts = {set_name:dict() for set_name in self.data_factory.dataset_names}

        labels_dict = {set_name:set() for set_name in self.data_factory.dataset_names}

        if isinstance(self.data_factory, deep_learning.data_factory_blueprints.ChunkFactory):
            for chunk_data_dict in self.data_factory.generate_all():
                for set_name, data_chunk in chunk_data_dict.items():
                    
                    class_matrix_dict = class_matrix_dict_of_dicts[set_name]
                    labels = labels_dict[set_name]

                    X_eval, y_eval = data_chunk

                    self._modify_labels_and_class_matrix_dict(X_eval, y_eval, labels, class_matrix_dict)
                
        elif isinstance(self.data_factory, deep_learning.data_factory_blueprints.MiniBatchFactory):
            mb_data_dict = self.data_factory.generate_all()
            for set_name, data in mb_data_dict.items():
                
                class_matrix_dict = class_matrix_dict_of_dicts[set_name]
                labels = labels_dict[set_name]

                X_eval, y_eval = data

                self._modify_labels_and_class_matrix_dict(X_eval, y_eval, labels, class_matrix_dict)
            
        report_items_dict = dict()

        for set_name in self.data_factory.dataset_names:
            labels = labels_dict[set_name]
            class_matrix_dict = class_matrix_dict_of_dicts[set_name]

            report_items_dict[set_name] = self._create_report_items(class_matrix_dict, labels)

        return report_items_dict

    def _modify_labels_and_class_matrix_dict(self, X_eval, y_eval, labels, class_matrix_dict):
        """predict the label in teh X_eval for the dataset and add the (true_label, pred_label) count in the class_matrix_dict,
        if there is a new label then add to labels set
        
        Args:
            X_eval (numpy array) : chunk/batch of examples to be evaluated
            y_eval (numpy array) : chunk/batch of labels for X_eval
            labels (set) : set of unique labels seen in the class_matrix_dict
            class_matrix_dict (dict) : (true_label, pred_label) count dictionary
        """
        y_pred_eval = self.model.predict_labels(X_eval)

        if not isinstance(self.model.loss_layer.loss_func, node_funcs.BCE):
            y_eval = np.argmax(y_eval, axis=1)

        for true_label, pred_label in zip(y_eval.ravel(), y_pred_eval.ravel()):
            true_label = int(true_label)
            pred_label = int(pred_label)

            class_matrix_dict[(true_label, pred_label)] = class_matrix_dict.get((true_label, pred_label), 0) + 1

            labels.add(true_label)
            labels.add(pred_label)

    def _create_report_items(self, class_matrix_dict, labels):
        sorted_labels_key = self._create_sorted_labels_key(labels)
        class_matrix = self._create_class_matrix(class_matrix_dict, sorted_labels_key)
        f1s = self._create_f1s(class_matrix)

        return sorted_labels_key, class_matrix, f1s

    @staticmethod
    def _create_sorted_labels_key(labels):
        sorted_labels = sorted(list(labels))
        sorted_labels_key = {label: idx for idx, label in enumerate(sorted_labels)}

        return sorted_labels_key
    
    @staticmethod
    def _create_class_matrix(class_matrix_dict, sorted_labels_key):

        num_labels = len(sorted_labels_key)

        class_matrix = np.zeros((num_labels, num_labels), dtype=int)

        for true_label, pred_label in class_matrix_dict:
            pair_count = class_matrix_dict[(true_label, pred_label)]
            class_matrix_idx = (sorted_labels_key[true_label], sorted_labels_key[pred_label])

            class_matrix[class_matrix_idx] = pair_count
        
        return class_matrix

    def _create_f1s(self, class_matrix):
        precisions = class_matrix.diagonal() / class_matrix.sum(axis=0)
        recalls = class_matrix.diagonal() / class_matrix.sum(axis=1)
        f1s = (2 * precisions * recalls) / (precisions + recalls + self._stable_constant)

        return f1s
    
    def save_driver(self, path):
        joblib.dump(self, path)

    def load_driver(path):
        potential_manager= joblib.load(path)
        return potential_manager