# data factories transform raw data and generate data ready for your models

from collections import OrderedDict
import csv
import numpy as np
import queue
import threading
import warnings
from . import utils, no_resources
import pandas as pd

# TODO

# fix how output dim and input dim are calculated (has to be after transform), maybe can use infer as well
# fix creating linked chunks for MiniBatches where you load in data directly
# Pull transformations out of data-factory
# handle standardization (or just depend on someone calculating mean std outside of the )


# COMPLETED
# extracted data transformation (except for standardize)


class DataFactoryFoundation:
    """Parent class to share functionality between other generators. 
    Generators handle reading data from files, transforming the data if needed, and returning the transformed data 
    (ideally for statistical model training).

    Training examples are located in and "input" file while their labels are located in an "output" file (input and output files can be the same file)

    Attributes:
        batch_size (int) : size of the batch to be generated
        _input_flag (bool) : if input props have been set
        _output_flag (bool) : if output props have been set

        input_path (str) : path to the input file
        _input_jar (utils.JarOpener) : jar opener for the input path
        _data_input_selector (numpy.IndexExpression) : 1D index expression to select certain columns
        _input_skiprows (int) : number of rows to skip in input file before starting to read
        _standardize (bool) : whether or not to standardize the input data

        output_path (str) : path to the output file
        _output_jar (utils.JarOpener) : jar opener for the output path
        _data_output_selector (numpy.IndexExpression) : 1D index expression to select certain columns
        _output_skiprows (int) : number of rows to skip in the output file before starting to read
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

        self._input_flag = False
        self._output_flag = False

        self._standardize = False


    # setting data properties
    def set_data_input_props(self, input_path, data_selector=np.s_[:], skiprows=0, standardize=False, input_transform=None):
        """Set the data/input properties for the chunk object

        Args:
            input_path (str) : path for input data
            data_selector (IndexExpression, default=np.s_[:]) : 1D index expression to select certain columns, if none specified will select all columns
            skiprows (int, default=0) : number of rows to skip
            standardize (bool, default=False) : whether or not to standardize data
            input_transform (function, default=None) : transform function for the input data
        """
        # mark input has been set
        self._input_flag=True

        # get the opener function
        self.input_path = input_path
        self._input_jar = utils.JarOpener(input_path)

        # select all columns if no data columns
        self._data_input_selector = data_selector

        self._input_skiprows = skiprows

        self.input_transform = input_transform

        ###
        self._standardize = standardize
        ###
        
        self._input_blank_flag = self._set_has_blank_flag(self._input_jar)

        if not self._input_blank_flag:
            self.set_input_dim()

        self._set_num_data_lines()
        self._num_chunks = self._num_data_lines / self.batch_size

    def _set_num_data_lines(self):
        """Retrieve total length of the data """
        with self._input_jar as data_file:
            for _ in range(self._input_skiprows):
                next(data_file)
            data_lines = sum(1 for line in data_file)

        self._num_data_lines = data_lines

    def _set_has_blank_flag(self, jar):
        """Sets if there are none/blank inputs in the data"""
        with jar as data_file:
            reader = csv.reader(data_file)
            return any("" in line for line in reader)

    def set_data_output_props(self, output_path, data_selector=np.s_[:], skiprows=0, output_transform=None):
        """Set the label properties for the chunk object

        Args:
            output_path (str) : path for output data
            data_selector (IndexExpression, default=np.s_[:]) : 1D index expression to select certain columns, if none specified will select all columns
            skiprows (int, default=0) : number of rows to skip
            output_transform (function, default=None) : transform function for the input data
        """

        self._output_flag = True

        # get the opener function
        self.output_path = output_path
        self._output_jar = utils.JarOpener(output_path)

        self._data_output_selector = data_selector

        self._output_skiprows = skiprows

        self.output_transform = output_transform

        # handle one hot encoding
        self._output_blank_flag = self._set_has_blank_flag(self._output_jar)

        if not self._output_blank_flag:
            self.set_output_dim()

        self._output_flag = True

    # Data property 

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self):
        raise Exception("Cannot directly set input_dim")

    def set_input_dim(self):
        """Set the dimension of the input data by peeking inside the input data file"""
        dim = self.get_selector_dim(self._input_jar, self._data_input_selector, self._input_skiprows, self.input_transform)

        self._input_dim = dim

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self):
        raise Exception("Cannot directly set output_dim")

    def set_output_dim(self):
        """Set the output dimension (after one hot encoding)"""
        dim = self.get_selector_dim(self._output_jar, self._data_output_selector, self._output_skiprows, self.output_transform)

        self._output_dim = dim

    @staticmethod
    def get_selector_dim(jar_opener, data_selector, skip_lines, transform_func):
        """Return the dimension of selected data within file"""
        with jar_opener as file:
            reader = csv.reader(file)
            for _ in range(skip_lines):
                next(reader)
            line = np.array(next(reader), dtype=float)[data_selector]
            if transform_func is not None:
                line = transform_func(line)
                selector_dim = line.shape[-1]
            else:
                selector_dim = len(line)

        return selector_dim

class MiniBatchAssembly(DataFactoryFoundation):
    """Generator to generate mini batches. loads data into memory before generating the data

    New Attributes (See parent class PreDataGenerator for the inherited attributes from __init__):

        _train_generator (bool) : if the generator is the train generator or not
        _linked_generator (bool) : if the generator is linked to another generator and shares properties with that other generator

        X (numpy.ndarray) : design matrix, input data
        y (numpy.ndarray) : label matrix/array, output data
    """
    def __init__(self, batch_size, train_generator=False):
        super().__init__(batch_size)

        self._train_generator = train_generator
        self._linked_generator = False

        self.X = None
        self.y = None
        self._data_loaded = False

    # property setters

    def set_data_input_props(self, input_path, data_selector=np.s_[:], skiprows=0, standardize=False, input_transform=None):
        """see parent class, additionally loads data if input flag and output flag are True"""
        super().set_data_input_props(input_path, data_selector, skiprows, standardize, input_transform)

        if self._input_flag and self._output_flag:
            self.load_file_data()

    def set_data_output_props(self, output_path, data_selector=np.s_[:], skiprows=0, output_transform=None):
        """see parent class, additionally loads data if input flag and output flag are True"""
        super().set_data_output_props(output_path, data_selector, skiprows, output_transform)
        if self._input_flag and self._output_flag:
            self.load_file_data()

    # Data generation
    def generate_input_data(self):
        """Generate X_data input"""
        return self.generate_raw_jar_data(self._input_flag, self._input_jar, self._input_skiprows, self._data_input_selector, 0, self.input_transform, self._input_blank_flag)

    def generate_output_data(self):
        """Generate X_data input"""
        return self.generate_raw_jar_data(self._output_flag, self._output_jar, self._output_skiprows, self._data_output_selector, 2, self.output_transform, self._output_blank_flag)

    def generate_raw_jar_data(self, set_flag, jar_opener, skiprows, data_selector, ndmin, transform_func, blank_flag):
        """Generate raw jar data without transforming the data

        Args:
            set_flag (bool) : whether or not properties for input/output have been set
            jar_opener (utils.JarOpener) : jar opener of file to open
            skiprows (int) : number or rows to skip
            data_selector (IndexExpression) : numpy index expression to select data from generated raw data
            ndmin (int) : The returned array will have at least ndmin dimensions. Otherwise mono-dimensional axes will be squeezed. 
                Legal values: 0 (default), 1 or 2.
            transform_func (function) : Function to transform the data in a certain way
            blank_flag (bool) : if the file has None/blank inputs

        Yields:
            data (numpy array) : data from jar opener of set chunk size
        """

        # check if input and output are set
        if not set_flag:
            raise Exception("Please set data input properties (set_data_input_props)")

        # open the file
        with jar_opener as data_file:
            # skip rows
            for _ in range(skiprows):
                next(data_file)

            if blank_flag:
                data = pd.read_csv(data_file)
            else:
                # read data
                data = np.loadtxt(data_file, delimiter=",", ndmin=ndmin, dtype=int)

            data = data[:,data_selector]

            if transform_func is not None:
                data = transform_func(data)

        return data
    
    def load_file_data(self):
        X_data, y_data = self.generate_input_data(), self.generate_output_data()
        self.load_data(X_data, y_data)

    def load_data(self, X_data, y_data):
        """Loads transformed data into memory for generation
        
        Args:
            X_data (numpy array) : possible X/input data
            y_data(numpy array) : possible y/output data
        """

        # validate that the jars link up
        if len(X_data) != len(y_data):
            raise Exception("Input file and output file are not the same length")

        if self._standardize:
            if self._train_generator:
                self._train_mean = X_data.mean(axis=0)
                self._train_std = X_data.std(axis=0)

            if self._train_generator or self._linked_generator:
                #X_data = (X_data - self._train_mean) / self._train_std
                X_data = (X_data - 33.3183) / 78.567

        self.X, self.y = X_data, y_data

        self._data_loaded = True

    def generate(self):
        """Generates input and output data

        Returns:
            X_train_batch (numpy.ndarray) : (num_examples, ...) for X data
            y_train_batch (numpy.ndarray) : (num_examples, ...) for y data
        """
        if not self._data_loaded:
            self.load_file_data()

        m = len(self.X)

        for start_idx in range(0,m-1,self.batch_size):
            end_idx = min(start_idx + self.batch_size, m)

            # locate relevant fields to 
            X_train_batch = self.X[start_idx:end_idx]
            y_train_batch = self.y[start_idx:end_idx]

            yield X_train_batch, y_train_batch

    def create_linked_gen_data(self, X, y):
        linked_chunk = MiniBatchAssembly(batch_size=self.batch_size, train_generator=False)
        linked_chunk._linked_generator = True
        linked_chunk.load_data(X,y)
        return linked_chunk


    def create_linked_generator(self, input, output):
        """Create a chunk linked to the instance train chunk, ex. a dev or test chunk

        Args:
            input_path (str) : path for input data
            output_path (str) : path for output data
            # generator_type (no_resources generator) : generator
        Returns:
            linked_chunk (PreDataGenerator) : linked chunk to the train chunk
        """
        input_jar = utils.JarOpener(input)
        output_jar = utils.JarOpener(output)
        self.val_create_linked_generator(input_jar, output_jar)

        linked_chunk = MiniBatchAssembly(batch_size=self.batch_size, train_generator=False)
        linked_chunk._linked_generator = True

        # set data and properties
        linked_chunk.set_data_input_props(input_path=input,
                                        data_selector=self._data_input_selector,
                                        skiprows=self._input_skiprows,
                                        standardize=self._standardize,
                                        input_transform=self.input_transform)

        if self.input_dim != linked_chunk.input_dim:
            raise Exception("Input data dimensions are not the same")

        linked_chunk.set_data_output_props(output_path=output,
                                        data_selector=self._data_output_selector,
                                        skiprows=self._output_skiprows,
                                        output_transform=self.output_transform)
        if self.output_dim != linked_chunk.output_dim:
            raise Exception("Output data dimensions are not the same")

        if self._standardize:
            linked_chunk._train_mean = self._train_mean
            linked_chunk._train_std = self._train_std

        return linked_chunk

    def val_create_linked_generator(self, input_chunk, output_chunk):
        """Validate train chunk"""
        # validation
        if not self._train_generator:
            raise Exception("Can only create linked generators from a train generator")
        if not self._input_flag:
            raise Exception("Must set input properties for train generator before linking")
        if not self._output_flag:
            raise Exception("Must set output properties for train generator before linking")

class MiniBatchFactory:
    """Object to handle passing mini-batches to a model

    Attributes:
        name_data_dict (dict) : {dataset_name : MiniBatchAssembly} dict
        train_key (str) : key string for the training mini-batch generator
        dev_key (str, default=None) : key string for the dev mini-batch generator

        train_generator (MiniBatchGenerator) : 
        dev_generator (MiniBatchGenerator) :
    """

    def __init__(self, name_data_dict, train_key, dev_key=None):
        """
        """
        self.name_data_dict = name_data_dict
        self.dataset_names = list(name_data_dict.keys())

        self.train_generator = self.name_data_dict[train_key]

        if dev_key:
            self.dev_generator = self.name_data_dict[dev_key]
        else:
            self.dev_generator = None

    def generate_all(self):
        """Generate data for all datasets and return in a dictionary"""

        generator_dict = dict()

        for dataset_name, generator in self.name_data_dict.items():

            generator_dict[dataset_name] = generator.X, generator.y

        return generator_dict

    # Properties and validation
    @property
    def train_key(self):
        return self._train_key

    @train_key.setter
    def train_key(self, train_key_cand):

       self.val_key(train_key_cand)

       self._train_key = train_key_cand

    @property
    def dev_key(self):
        return self._dev_key

    @dev_key.setter
    def dev_key(self, dev_key_cand):

        if dev_key_cand is not None:
            self.val_key(dev_key_cand)

        self._dev_key = dev_key_cand

    def val_key(self, key_name):

        if key_name not in self.name_data_dict.keys():
            raise Exception(f"{key_name} not in data split keys")

class ChunkFactory(DataFactoryFoundation):
    """Object to handle passing chunks to a model

    Handles ChunkManagerChild objects 

    Attributes:

    """
    def __init__(self,
                 batch_size,
                 data_split,
                 train_key,
                 dev_key=None,
                 seed=100):

        super().__init__(batch_size=batch_size)

        self.data_split = data_split
        self.dataset_names = list(self.data_split.keys())

        self.train_key = train_key
        self.dev_key = dev_key

        self.val_data_split()

        self._set_train_and_dev_children()

        self.seed = seed

    def set_data_input_props(self, input_path, data_selector=np.s_[:], skiprows=0, standardize=False, input_transform=None):

        super().set_data_input_props(input_path, data_selector, skiprows, standardize, input_transform=input_transform)
        # calculate mean and standard deviation of training data if standardizing

        if self._standardize:
            self._set_training_data_mean()
            self._set_training_data_std()

    # Properties and validation
    @property
    def train_key(self):
        return self._train_key

    @train_key.setter
    def train_key(self, train_key_cand):

       self.val_key(train_key_cand)

       self._train_key = train_key_cand

    @property
    def dev_key(self):
        return self._dev_key

    @dev_key.setter
    def dev_key(self, dev_key_cand):

        if dev_key_cand is not None:
            self.val_key(dev_key_cand)

        self._dev_key = dev_key_cand

    def val_key(self, key_name):

        if key_name not in self.data_split.keys():
            raise Exception(f"{key_name} not in data split keys")

    @property
    def data_split(self):
        return self._data_split

    @data_split.setter
    def data_split(self, data_split_tuple_cand):
        """Validate and set data_split"""

        self._data_split = data_split_tuple_cand

    def val_data_split(self):
        # validation
        val_list = np.array(list(self.data_split.values()))

        train_share = self.data_split[self.train_key]

        if np.any(val_list < 0) or np.any(val_list > 1):
            raise AttributeError("Data splits must be between 0 and 1 inclusive")

        if train_share == 0:
            raise AttributeError("Training share must be greater than 0")

        if not np.isclose(val_list.sum(), 1):
            raise AttributeError("Data split must sum to 1")

    def _set_train_and_dev_children(self):

        self.train_generator = self.create_and_get_child(self.train_key)
        self.train_generator._train_generator = True
        if self.dev_key is not None:
            self.dev_generator = self.create_and_get_child(self.dev_key)
        else:
            self.dev_generator = None

    # statistics

    def _set_training_data_mean(self):

        rng = np.random.default_rng(self.seed)

        train_sum = np.zeros(self.input_dim)
        train_count = 0

        for X_data in self.generate_input_data():
            data_len = X_data.shape[0]
            split_idxs = self._get_split_idxs(data_len, rng)[self.train_key]

            X_train = X_data[split_idxs]
            train_len = X_train.shape[0]

            train_sum += X_train.sum(axis=0)
            train_count += train_len

        self._train_mean = train_sum / train_count

    def _set_training_data_std(self):
        # set the seed

        rng = np.random.default_rng(self.seed)

        sum_dev_sqd = np.zeros(self.input_dim)
        train_count = 0

        for X_data in self.generate_input_data():
            data_len = X_data.shape[0]
            split_idxs = self._get_split_idxs(data_len, rng)[self.train_key]

            X_train = X_data[split_idxs]
            train_len = X_train.shape[0]

            sum_dev_sqd += ((X_train - self._train_mean) ** 2).sum(axis=0)
            train_count += train_len

        self._train_std = np.sqrt(sum_dev_sqd / train_count)

    # Data generation

    def generate_input_data(self):
        """Generate X_data input"""
        return self.generate_raw_jar_data(self._input_flag, self._input_jar, self._input_skiprows, self._data_input_selector, 0, self.input_transform, self._input_blank_flag)

    def generate_output_data(self):
        """Generate X_data input"""
        return self.generate_raw_jar_data(self._output_flag, self._output_jar, self._output_skiprows, self._data_output_selector, 2, self.output_transform, self._output_blank_flag)

    def generate_raw_jar_data(self, set_flag, jar_opener, skiprows, data_selector, ndmin, transform_func, blank_flag):
        """Generate raw jar data without transforming into no_resources.OneHotArray for input or one hot vector for output data

        Args:
            set_flag (bool) : whether or not properties for input/output have been set
            jar_opener (utils.JarOpener) : jar opener of file to open
            skiprows (int) : number or rows to skip
            data_selector (IndexExpression) : numpy index expression to select data from generated raw data
            ndmin (int) : The returned array will have at least ndmin dimensions. Otherwise mono-dimensional axes will be squeezed. 
                Legal values: 0 (default), 1 or 2.
            blank_flag (bool) : if the file has None/blank inputs

        Yields:
            data (numpy array) : data from jar opener of set chunk size
        """

        # check if input and output are set
        if not set_flag:
            raise Exception("Please set data input properties (set_data_input_props)")

        # open the file
        with jar_opener as data_file:

            # skip rows
            for _ in range(skiprows):
                next(data_file)

            # obtain the chunk of X_data
            if blank_flag: # to read in None values
                for chunk in pd.read_csv(data_file, chunksize=self.batch_size):
                    data = chunk.to_numpy()
                    data = data[:,data_selector]

                    if transform_func is not None:
                        data = transform_func(data)

                    yield data
            else:
                while True:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data = np.loadtxt(data_file, delimiter=",", max_rows=self.batch_size, ndmin=ndmin, dtype=int)

                    data_len = data.shape[0]

                    # split the data into training data
                    if data_len == 0:
                        break

                    data = data[:,data_selector]

                    if transform_func is not None:
                        data = transform_func(data)

                    yield data

    def build_queue(self, q_to_build, generator):
        """Target function for thread

        Args:
            q_to_build (queue.Queue) : queues to store data from generator
            generator (PreDataGenerator) : some type fo generator for io process 
        """
        for data in generator:
            q_to_build.put(data)
        q_to_build.put(None)

    def generate_raw_data(self):
        """threaded generation, minimal speed increase
        creates queues for input data and output data, io processes for fetching input data and output data can occur at the same time"""
        input_queue = queue.Queue(maxsize=10)
        output_queue = queue.Queue(maxsize=10)

        # daemon threads for reliable shutdown of threads in case of exception
        input_thread = threading.Thread(target=self.build_queue, args=(input_queue, self.generate_input_data()), daemon=True)
        output_thread = threading.Thread(target=self.build_queue, args=(output_queue, self.generate_output_data()), daemon=True)
        input_thread.start()
        output_thread.start()

        while True:
            X_data = input_queue.get()
            y_data = output_queue.get()

            if X_data is None and y_data is None:
                break

            if len(X_data) != len(y_data):
                raise Exception("Input file and output file are not the same length")

            if self._standardize:
                #X_data = (X_data - self._train_mean) / self._train_std
                X_data = (X_data - 33.3183) / 78.567

            yield X_data, y_data

        input_thread.join()
        output_thread.join()

    def generate(self, gen_key=None):

        if gen_key is None:
            gen_key = self.train_key

        rng = np.random.default_rng(self.seed)
        for X_data, y_data in self.generate_raw_data():

            split_idxs = self._get_split_idxs(X_data.shape[0], rng)[gen_key]

            if self._input_blank_flag or self._output_blank_flag:
                split_idxs.sort()

            X = X_data[split_idxs]
            y = y_data[split_idxs]

            yield X, y

    def generate_all(self):
        """Generate data for all datasets and return in a dictionary"""

        chunk_dict = OrderedDict()
        rng = np.random.default_rng(self.seed)
        for X_data, y_data in self.generate_raw_data():

            split_idx_dict = self._get_split_idxs(X_data.shape[0], rng)

            for set_name, idxs in split_idx_dict.items():
                chunk_dict[set_name] = (X_data[idxs], y_data[idxs])

            yield chunk_dict

    def _get_split_idxs(self, data_length, rng):
        """Retrieve indexes of different data sets for data of given length

        Args:
            data_length (int) : length of the data to retrieve idxs from
            rng (numpy.random.Generator) : generator to use for splitting idxs

        Returns:
            split_idxs (dict) : {set_name -> data index}
        """

        shuffled_idxs = rng.permutation(data_length)

        split_idxs = OrderedDict()

        beg_idx = 0
        for idx, set_name in enumerate(self.data_split):

            if idx == len(self.data_split) - 1:
                split_idxs[set_name] = shuffled_idxs[beg_idx:]
            else:
                share = self.data_split[set_name]
                end_idx = min(beg_idx + round(share * data_length), data_length)
                split_idxs[set_name] = shuffled_idxs[beg_idx:end_idx]

            beg_idx = end_idx

        return split_idxs

    def create_and_get_child(self, gen_key):
        """Create and retrieve a ChunkManagerChild object to pass to model functions

        Args:
            gen_key (str) : key string for the child (in data split)

        Returns:
            child (ChunkManagerChild) : created 
        """
        child = ChunkAssembly(self, gen_key)
        return child
    
class ChunkAssembly:
    """Class to act as an eval chunk like MiniBatchGenerator. To be passed into function to evaluation model performance.
    Ex. would create as a dev set with gen_key "dev", pass a ChunkManager with a data set called "dev", 

    Definitely a hack to smooth out functionality of the two nn classes"""

    def __init__(self, chunk_factory:ChunkFactory, gen_key):

        self._train_generator = False

        self.chunk_factory = chunk_factory

        self.gen_key = gen_key

        self.batch_size = self.chunk_factory.batch_size

    @property
    def gen_key(self):
        return self._gen_key

    @gen_key.setter
    def gen_key(self, gen_key_cand):

        self.chunk_factory.val_key(gen_key_cand)

        self._gen_key = gen_key_cand

    def generate(self):
        return self.chunk_factory.generate(self.gen_key)
