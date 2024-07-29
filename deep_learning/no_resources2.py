import os
import tempfile
import numpy as np
import csv
import gzip
import re
import time
import warnings
import traceback
import random
from contextlib import ExitStack
import queue 
import threading
from collections import OrderedDict

from . import utils


#TODO

## option for generating ints or floats (maybe in set data input and output)
## chunk sizes mean batch sizes for chunk nn, but batch_size * train_prop = batch size for super chunk
## oha dict allows for double indexing
## __iter__ and __next__ for generate
## apply one hot encoding for y_data just to OHA
## file extensions for jar should not come from name, but from actual file attributes


# COMPLETED
## automatically create jar openers for setting input


# REJECTED



## can put one hot labels in txt


# shuffle based on https://towardsdatascience.com/randomizing-very-large-datasets-e2b14e507725
def shuffle_in_memory(source, output, header_bool=False):
    """Shuffle lines from a file after reading the entire file into memory
    
    Args:
        source (file-like) : file-like source whose lines are to be shuffled
        output (file-like) : destination to write to
        header_bool (bool, default=False) : whether or not source file has header
    """
    with open(source) as sf:
        if header_bool:
            header = sf.readline()
        lines = sf.readlines()

    random.shuffle(lines)

    with open(output, "w") as of:
        if header_bool:
            of.write(header)
        of.writelines(lines)

def merge_files(temp_files, output, header=None):
    """Merge temp file lines into output file
    
    Args:
        temp_files (array) : array of temp_files to merge
        output (file-like) : destination to write temp_files to
        header (str, default=None) : header to include in output
    """

    with open(output, "w") as of:    
        if header:
            of.write(header)
        for temp_file in temp_files:
            with open(temp_file.name) as tf:
                line = tf.readline()
                while line:
                    of.write(line)
                    line = tf.readline()
            
def shuffle(source, output, memory_limit, file_split_count=10, header_bool=False):
    """Shuffle lines from large source file into output file without reading the entire source file into memory

    Args:
        source (file-like) : file-like source whose lines need to be shuffled
        output (file-like) : destination to write to
        memory_limit (int) : byte limit to shuffle in memory
        file_split_count (int, default=10) : number of temp files to create, relates to recursion depth
        header_bool (bool, default=False) : whether or not source file has header
    """

    header = None
    if os.path.getsize(source) < memory_limit:
        shuffle_in_memory(source, output, header_bool)
    else:
        with ExitStack() as stack:
            temp_files = [stack.enter_context(tempfile.NamedTemporaryFile("w+", delete=False)) for i in range(file_split_count)]
    
            sf = stack.enter_context(open(source))

            if header_bool:
                header = sf.readline()
            for line in sf: 
                random_file_idx = random.randint(0, len(temp_files) - 1)
                temp_files[random_file_idx].write(line)
        
        for temp_file in temp_files:
            shuffle(temp_file.name, temp_file.name, memory_limit, file_split_count, header_bool=False)

        merge_files(temp_files, output, header)

def chop_up_csv(source_path, split_dict, header=True, seed=100):
    """Section given data source into different sets, 
    multinomial draw for each line of data corresponding to file destination
    
    Args:
        source_path (str) : path of file to break up
        split_dict (dict) : {new file name -> output probability} dictionary, must sum to 1
        header (bool, default=True) : if source file contains header or not
        seed (int, default=100) : seed for multinomial draw
    """
    _val_chop_up_csv(source_path, split_dict)

    # create ordered key value pair 
    ordered = [(name, prob) for name, prob in split_dict.items()]
    ordered_names = [item[0] for item in ordered]
    ordered_probs = [item[1] for item in ordered]

    opener = JarOpener(source_path)
    target_dir = utils.get_file_dir(source_path)

    with ExitStack() as stack:

        orig_file = stack.enter_context(opener)
        orig_file_reader = csv.reader(orig_file)

        output_writers = [csv.writer(stack.enter_context(open(f"{target_dir}/{output_name}", "w", newline=""))) for output_name in ordered_names]

        if header:
            header = next(orig_file_reader)
            for writer in output_writers:
                writer.writerow(header)
        
        for row in orig_file_reader:
            assigned = np.random.multinomial(1, ordered_probs, size=1).argmax()
            output_writers[assigned].writerow(row)

def _val_chop_up_csv(source_path, split_dict):
    
    for file_name in split_dict.keys():
        if not isinstance(file_name, str):
            raise Exception("split_dict keys (file names) must be strings")

    val_array = np.array(list(split_dict.values()))

    if np.any(val_array < 0) or np.any(val_array > 1):
        raise Exception("Probs must be between 0 and 1 inclusive")

    if val_array.sum() != 1:
        raise Exception("Probs must sum to 1")

class JarOpener:
    """A file opener to read a file at a given path
    
    Attributes:
        _opener (function)) : opener function for file
        _open_kwargs (dict) : kwargs for the opener
        open_source (file-like) : open file object
    """

    def __init__(self, source_path) -> None:
        self._opener, self._opener_kwargs = self.get_opener_attrs(source_path)
    
    @staticmethod
    def get_file_extension(source_path):
        """Get the file extension of a file path
        
        Args:
            source_path (str) : file path string

        Returns:
            file_extension (str) : The file extension 
        """
        _, file_extension = os.path.splitext(source_path)
        return file_extension
        

    def get_opener_attrs(self, source_path):
        """Return the correct open function for the file extension
        
        Args:
            source_path (str) : file path string
        
        Returns:
            opener (function) : function to open file
            opener_kwargs (dict) : args for the opener function
        """
        file_extension = self.get_file_extension(source_path)

        if file_extension == ".csv":
            opener = open
            opener_kwargs = {"file": source_path, "mode": "r", "encoding": None}
        elif file_extension == ".gz":
            opener = gzip.open
            opener_kwargs =  {"filename": source_path, "mode": "rt", "encoding":"utf-8"}
        else:
            raise Exception("File extension not supported")
        
        return opener, opener_kwargs

    def __enter__(self):
        """built-in method to control entrance in context manage, returns that open file"""
        self.open_source = self._opener(**self._opener_kwargs)
        return self.open_source

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """built-in method to control behavior upon exiting context manager"""
        # close the file if open
        if self.open_source:
            self.open_source.close()
            self.open_source = None # deleting the open file object so chunk can be saved
        
        # handle exceptions other than Generator Exit
        if exc_type and exc_type != GeneratorExit:
            print(f"An exception occurred: {exc_value}")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        # return False if want to propagate exceptions outside of the context block (not supressed)
        return False   

# DATA GENERATORS

class PreDataGenerator:
    """Parent class to share functionality between other generators. 
    Generators handle reading data from files, transforming the data if needed, and returning the transformed data 
    (ideally for statistical model training).

    Training examples are located in and "input" file while their labels are located in an "output" file (input and output files can be the same file)

    Attributes:
        batch_size (int) : size of the batch to be generated
        _input_flag (bool) : if input props have been set
        _output_flag (bool) : if output props have been set

        input_path (str) : path to the input file
        _input_jar (JarOpener) : jar opener for the input path
        _data_input_selector (numpy.IndexExpression) : 1D index expression to select certain columns
        _sparse_dim (int) : dimensions of the sparse vectors
        _input_skiprows (int) : number of rows to skip in input file before starting to read
        _standardize (bool) : whether or not to standardize the input data

        output_path (str) : path to the output file
        _output_jar (JarOpener) : jar opener for the output path
        _data_output_selector (numpy.IndexExpression) : 1D index expression to select certain columns
        _one_hot_width (list) : number of categories for one hot encoding
        _output_skiprows (int) : number of rows to skip in the output file before starting to read
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

        self._input_flag = False
        self._output_flag = False

    # setting data properties
    def set_data_input_props(self, input_path, data_selector=np.s_[:], skiprows=0, sparse_dim=None, standardize=False):
        """Set the data/input properties for the chunk object
        
        Args:
            input_path (str) : path for input data
            data_selector (IndexExpression, default=np.s_[:]) : 1D index expression to select certain columns, if none specified will select all columns
            skiprows (int, default=0) : number of rows to skip
            sparse_dim (int, default=None) : dimensions of the sparse vectors, if applicable
            standardize (bool, default=False) : whether or not to standardize data
        """
        # mark input has been set
        self._input_flag=True

        # get the opener function
        self.input_path = input_path
        self._input_jar = JarOpener(input_path)

        # select all columns if no data columns
        self._data_input_selector = data_selector

        self._sparse_dim = sparse_dim

        self._input_skiprows = skiprows

        self.set_input_dim()

        self._standardize = standardize

        self._set_num_data_lines()
        self._num_chunks = self._num_data_lines / self.batch_size

    def _set_num_data_lines(self):
        """Retrieve total length of the data """
        with self._input_jar as data_file:
            for _ in range(self._input_skiprows):
                next(data_file)
            data_lines = sum(1 for line in data_file)

        self._num_data_lines = data_lines
        
    def set_data_output_props(self, output_path, data_selector=np.s_[:], skiprows=0, one_hot_width=None):
        """Set the label properties for the chunk object
        
        Args:
            output_path (str) : path for output data
            data_selector (IndexExpression, default=np.s_[:]) : 1D index expression to select certain columns, if none specified will select all columns
            skiprows (int, default=0) : number of rows to skip
            one_hot_width (list, default=None) : number of categories for one hot encoding
        """
        
        self._output_flag = True

        # get the opener function
        self.output_path = output_path
        self._output_jar = JarOpener(output_path)

        self._data_output_selector = data_selector

        # handle one hot encoding
        self.one_hot_width = one_hot_width

        self._output_skiprows = skiprows

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

        if self._sparse_dim:
            dim = self._sparse_dim
        else:
            dim = self.get_selector_dim(self._input_jar, self._data_input_selector)

        self._input_dim = dim

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self):
        raise Exception("Cannot directly set output_dim")
    
    def set_output_dim(self):
        """Set the output dimension (after one hot encoding)"""
        if self.one_hot_width:
            dim = self.one_hot_width
        else:
            dim = self.get_selector_dim(self._output_jar, self._data_output_selector)

        self._output_dim = dim

    @staticmethod
    def get_selector_dim(jar_opener, data_selector):
        """Return the dimension of selected data within file"""
        with jar_opener as file:
            reader = csv.reader(file) 
            line = np.array(next(reader))
            selector_dim = len(line[data_selector])

        return selector_dim

    @property
    def one_hot_width(self):
        return self._one_hot_width
    
    @one_hot_width.setter
    def one_hot_width(self, one_hot_width_cand):
        """Validate one hot encoding for set_data_output_props and set if appropriate
        
        Args:
            one_hot_width_cand (int) : candidate for one_hot_width
        """
        if one_hot_width_cand is None:
            self._one_hot_width = one_hot_width_cand
        elif isinstance(one_hot_width_cand, int):
            dim = self.get_selector_dim(self._output_jar, self._data_output_selector)

            if dim == 1:
                self._one_hot_width = one_hot_width_cand
            else:
                raise AttributeError("Cannot set one_hot_width and one hot encode if dimensions of raw output is not 1")
        else:
            return AttributeError(f"one_hot_width must be an integer or None, value of {one_hot_width_cand} given")

    def one_hot_labels(self, y_data):
        """One hot labels from y_data
        
        Args:
            y_data (numpy array)

        Returns:
            one_hot_labels
        
        """
        one_hot_labels = np.zeros((y_data.size, self._one_hot_width))
        one_hot_labels[np.arange(y_data.size), y_data.astype(int).flatten()] = 1

        return one_hot_labels

    # Data Generation

    

class MiniBatchGenerator(PreDataGenerator):
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

    def set_data_input_props(self, input_path, data_selector=np.s_[:], skiprows=0, sparse_dim=None, standardize=False):
        """see parent class, additionally loads data if input flag and output flag are True"""
        super().set_data_input_props(input_path, data_selector, skiprows, sparse_dim, standardize)
        
        if self._input_flag and self._output_flag:
            self._load_data()

    def set_data_output_props(self, output_path, data_selector=np.s_[:], skiprows=0, one_hot_width=None):
        """see parent class, additionally loads data if input flag and output flag are True"""
        super().set_data_output_props(output_path, data_selector, skiprows, one_hot_width)
        if self._input_flag and self._output_flag:
            self._load_data()

    # Data generation
    def generate_input_data(self):
        """Generate X_data input"""
        return self.generate_raw_jar_data(self._input_flag, self._input_jar, self._input_skiprows, self._data_input_selector)

    def generate_output_data(self):
        """Generate X_data input"""
        return self.generate_raw_jar_data(self._output_flag, self._output_jar, self._output_skiprows, self._data_output_selector, ndmin=2)

    def generate_raw_jar_data(self, set_flag, jar_opener, skiprows, data_selector, ndmin=0):
        """Generate raw jar data without transforming the data
        
        Args:
            set_flag (bool) : whether or not properties for input/output have been set
            jar_opener (JarOpener) : jar opener of file to open
            skiprows (int) : number or rows to skip
            data_selector (IndexExpression) : numpy index expression to select data from generated raw data
            ndmin (int, default=0) : The returned array will have at least ndmin dimensions. Otherwise mono-dimensional axes will be squeezed. 
                Legal values: 0 (default), 1 or 2.

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

            # read data
            data = np.loadtxt(data_file, delimiter=",", ndmin=ndmin, dtype=int)

            data = data[:,data_selector]

        return data

    def _load_data(self):
        """Loads transformed data into memory for generation"""
        X_data, y_data = self.generate_input_data(), self.generate_output_data()

        # validate that the jars link up
        if len(X_data) != len(y_data):
            raise Exception("Input file and output file are not the same length")

        if self._sparse_dim:
            X_data = OneHotArray(shape=(len(X_data),self._sparse_dim),idx_array=X_data)

        if self._one_hot_width:
            y_data = self.one_hot_labels(y_data)

        if self._standardize:
            if self._sparse_dim is not None:
                raise Exception("Generator does not support standardization for sparse dim")
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
            self._load_data()
        
        m = len(self.X)

        for start_idx in range(0,m-1,self.batch_size):
            end_idx = min(start_idx + self.batch_size, m)
            
            # locate relevant fields to 
            X_train_batch = self.X[start_idx:end_idx]
            y_train_batch = self.y[start_idx:end_idx]

            yield X_train_batch, y_train_batch
    
    def create_linked_generator(self, input_path, output_path):
        """Create a chunk linked to the instance train chunk, ex. a dev or test chunk
        
        Args:
            input_path (str) : path for input data
            output_path (str) : path for output data
            # generator_type (no_resources generator) : generator
        Returns:
            linked_chunk (PreDataGenerator) : linked chunk to the train chunk
        """
        input_jar = JarOpener(input_path)
        output_jar =  JarOpener(output_path)
        self.val_create_linked_generator(input_jar, output_jar)

        linked_chunk = MiniBatchGenerator(batch_size=self.batch_size, train_generator=False)
        linked_chunk._linked_generator = True

        # set data and properties
        linked_chunk.set_data_input_props(input_path=input_path,
                                        data_selector=self._data_input_selector, 
                                        skiprows=self._input_skiprows, 
                                        sparse_dim=self._sparse_dim, 
                                        standardize=self._standardize)
        
        if self.input_dim != linked_chunk.input_dim:
            raise Exception("Input data dimensions are not the same")

        linked_chunk.set_data_output_props(output_path=output_path,
                                        data_selector=self._data_output_selector,
                                        skiprows=self._output_skiprows,
                                        one_hot_width=self.one_hot_width)
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

class MiniBatchManager():
    """Object to handle passing mini-batches to a model
    
    Attributes:
        name_data_dict (dict) : {dataset_name : MiniBatchGenerator} dict
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

    
class ChunkManager(PreDataGenerator):
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

    def set_data_input_props(self, input_path, data_selector=np.s_[:], skiprows=0, sparse_dim=None, standardize=False):
        
        super().set_data_input_props(input_path, data_selector, skiprows, sparse_dim, standardize)
        # calculate mean and standard deviation of training data if standardizing
        
        if self._standardize:
            if self._sparse_dim is not None:
                raise Exception("Generator does not support standardization for sparse dim")
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
        return self.generate_raw_jar_data(self._input_flag, self._input_jar, self._input_skiprows, self._data_input_selector)

    def generate_output_data(self):
        """Generate X_data input"""
        return self.generate_raw_jar_data(self._output_flag, self._output_jar, self._output_skiprows, self._data_output_selector, ndmin=2)

    def generate_raw_jar_data(self, set_flag, jar_opener, skiprows, data_selector, ndmin=0):
        """Generate raw jar data without transforming into OneHotArray for input or one hot vector for output data
        
        Args:
            set_flag (bool) : whether or not properties for input/output have been set
            jar_opener (JarOpener) : jar opener of file to open
            skiprows (int) : number or rows to skip
            data_selector (IndexExpression) : numpy index expression to select data from generated raw data
            ndmin (int, default=0) : The returned array will have at least ndmin dimensions. Otherwise mono-dimensional axes will be squeezed. 
                Legal values: 0 (default), 1 or 2.

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
            while True:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = np.loadtxt(data_file, delimiter=",", max_rows=self.batch_size, ndmin=ndmin, dtype=int)

                data_len = data.shape[0]

                # split the data into training data
                if data_len == 0:
                    break

                data = data[:,data_selector]
                
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
            
            if self._sparse_dim:
                X_data = OneHotArray(shape=(len(X_data),self._sparse_dim),idx_array=X_data)
 
            if self._one_hot_width:
                y_data = super().one_hot_labels(y_data)

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

            # if batch_num == 1:
            #     print(dev_idxs)

            X = X_data[split_idxs]
            y = y_data[split_idxs]

            yield X, y
            #batch_num += 1

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
        child = ChunkManagerChild(self, gen_key)
        return child

class ChunkManagerChild:
    """Class to act as an eval chunk like MiniBatchGenerator. To be passed into function to evaluation model performance.
    Ex. would create as a dev set with gen_key "dev", pass a ChunkManager with a data set called "dev", 
    
    Definitely a hack to smooth out functionality of the two nn classes"""

    def __init__(self, chunk_manager:ChunkManager, gen_key):
        
        self._train_generator = False

        self.chunk_manager = chunk_manager

        self.gen_key = gen_key

        self.batch_size = self.chunk_manager.batch_size

    @property
    def gen_key(self):
        return self._gen_key
    
    @gen_key.setter
    def gen_key(self, gen_key_cand):

        self.chunk_manager.val_key(gen_key_cand)

        self._gen_key = gen_key_cand
    
    def generate(self):
        return self.chunk_manager.generate(self.gen_key)


class OneHotArray:
    """Sparse array for maximizing storage efficiency

    Attributes:

    """
    def __init__(self, shape, idx_array=None, oha_dict=None):
        """
        Args:
            shape (tuple) : dimensions of the array
            idx_array (array, default=None) : array where each row corresponds to a row vector in the OneHotArray
                integers in the array correspond to column indices of the 1 entries, 
                only positive integers allowed, except for -1 which counts as null space
            oha_dict (dict) : {row:col_idxs} dict

        """
        self.shape = shape
        self.ndim = 2

        # instantiate cand_idx_rel dict to hold sparse array
        cand_idx_rel = {}

        if isinstance(idx_array, (np.ndarray, list)) and oha_dict == None:
            if self.shape[0] < len(idx_array):
                raise Exception("Number of row vectors in array must be greater than amount given")
            for row_idx, col_idxs in enumerate(idx_array):
                filtered_col_idxs = self.filter_col_idxs(col_idxs)
                cand_idx_rel[row_idx] = filtered_col_idxs

        elif oha_dict != None and idx_array == None:
            
            if oha_dict.keys():
                if self.shape[0] < max(oha_dict.keys()) + 1:
                    raise Exception("Number of row vectors in array must be greater than max row index plus one")
                
            for row_idx, col_idxs in oha_dict.items():
                self.validate_idx(row_idx, axis=0)
                filtered_col_idxs = self.filter_col_idxs(col_idxs)
                cand_idx_rel[row_idx] = filtered_col_idxs
        else:
            raise Exception("Must either instantiate OneHotArray with an idx_array or oha_dict")

        self.idx_rel = cand_idx_rel
    
    def filter_col_idxs(self, raw_col_idxs):
        """Add valid column idxs to list and return valid idxs (in range of reference matrix when 0-indexed)

        Args:
            raw_col_idxs (array-like): list of possible column idxs
        
        Returns: 
            filtered_col_idxs (list): valid col idxs for a given row
        """
        filtered_col_idxs = []
        for col_idx in raw_col_idxs:
            if col_idx >= 0:
                self.validate_idx(col_idx, axis=1)
                filtered_col_idxs.append(int(col_idx))
            if col_idx < -1:
                raise Exception("No negative indices allowed (besides -1 which represents null space)")

        return filtered_col_idxs

    def to_array(self):

        array = np.zeros(self.shape)

        for row_idx, col_idxs in self.idx_rel.items():
            for col_idx in col_idxs:
                array[row_idx, col_idx] = 1

        return array
    
    def __matmul__(self, other):
        
        # validation
        if other.ndim != 2:
            raise Exception("Dimensions of composite transformations must be 2")

        if isinstance(other, np.ndarray): #sparse - dense multiplication
            # validation
            if self.shape[1] != other.shape[0]:
                raise Exception("Inner dimensions must match")
            outside_dims = (self.shape[0], other.shape[1])
            # qualify Row Sparse Array
            if len(self.idx_rel) < .5 * self.shape[0]:
                row_idxs = []
                product = np.zeros((len(self.idx_rel), other.shape[1]))

                counter = 0
                for row_idx, col_idxs in self.idx_rel.items():
                    row_idxs.append(row_idx)
                    product[counter] = other[col_idxs].sum(axis=0)
                    counter+=1

                return RowSparseArray(row_idx_vector=np.array(row_idxs), 
                                      dense_row_array=product,
                                      total_array_rows=self.shape[0])
            else:
                product = np.zeros(outside_dims)

                for row_idx, col_idxs in self.idx_rel.items():
                    product[row_idx] = other[col_idxs].sum(axis=0)

            return product
        
        elif isinstance(other, OneHotArray):
            return NotImplemented
        else:
            raise Exception("OneHotArray can only matrix multiply with numpy array or another OneHotArray")

    def __rmatmul__(self, other):
        # (b x s) (s x next layer) will become O(b) 
        # validation
        if other.ndim != 2:
            raise Exception("Dimensions of composite transformations must be 2")
        
        if isinstance(other, np.ndarray): # dense-sparse multiplication
            outside_dims = (other.shape[0], self.shape[1])

            product = np.zeros(outside_dims)
            
            if other.shape[1] != self.num_vectors:
                raise Exception("Inner dimensions must match")
            
            transposed = self.T

            for row_idx, col_idxs in transposed.idx_rel:
                product[:,row_idx] = other[:,col_idxs]

            return product
        elif isinstance(other, OneHotArray):
            pass
        else:
            raise Exception("OneHotArray can only matrix multiply with numpy array")

    def __getitem__(self, key):
        """Defining getitem to duck type with numpy arrays for 0th axis slicing and indexing"""
        # define dimensions and n_rows placeholder
        n_rows = 0
        n_cols = self.shape[1]
        
        gathered = {}
        if isinstance(key, int):
            n_rows = self.add_int_key(key, gathered, n_rows)
        elif isinstance(key, slice):
            n_rows = self.add_slice_key(key, gathered, n_rows)
        elif isinstance(key, (list, np.ndarray)):
            for sub_key in key:
                if isinstance(sub_key, tuple([int] + np.sctypes["int"])):
                    n_rows = self.add_int_key(sub_key, gathered, n_rows)
                else:
                    raise SyntaxError
        else:
            raise SyntaxError
        
        # for empty 
        if n_rows == 0:
            n_cols = 0

        return OneHotArray(shape=(n_rows,n_cols), oha_dict=gathered)
        
    def add_int_key(self, int_idx, gathered, n_rows):
        """Get integer index value 
        Args:
            int_idx (int) : integer row idx of the oha
            gathered (dict) : current gathered values of the indexed oha
            n_rows (int) : counter for amount of rows 

        Returns:
            n_rows (int) : number of rows in new oha
        """
        self.validate_idx(int_idx)
        if int_idx < 0:
            int_idx = self.convert_neg_idx(int_idx)
        # only need to gather rows in oha, 
        # if not in oha array (all zeroes) then should not be part of oha
        if int_idx in self.idx_rel:
            gathered[n_rows] = self.idx_rel[int_idx]
        
        n_rows += 1

        return n_rows

    def convert_neg_idx(self, idx):
        """Converts negative idxs for __getitem__ to positive idx
        Args:
            idx (int) : negative int to convert to positive
        """
        return self.shape[0] + idx
    
    def validate_idx(self, idx, axis=0):
        """See if the idx is out of bounds or not
        
        Args:
            idx (int) :  index to validate
            axis (int, default=0)
        """
        indexed_rows = self.shape[axis]
        if idx < -indexed_rows or idx > indexed_rows-1:
            raise IndexError(f"Given index {idx} does not exist")
        
    def add_slice_key(self, slice_obj, gathered, n_rows):
        """Add corresponding valid index values in slice to gathered
        
        Args:
            slice (slice) : key slice object
            gathered (dict) : current gathered values of the indexed oha
            n_rows (int) : counter for amount of rows 
        
        """
        if slice_obj.step is None:
            step = 1
        else:
            step = slice_obj.step
        for idx in range(slice_obj.start, slice_obj.stop, step):
            n_rows = self.add_int_key(idx, gathered, n_rows)

        return n_rows
    
    @property
    def T(self):
        """create a transpose of the one-hot array"""

        transpose_idx_rel = {}

        newshape = (self.shape[1], self.shape[0])

        for row_idx, row in self.idx_rel.items():
            for col_idx in row:
                if col_idx in transpose_idx_rel:
                    transpose_idx_rel[col_idx].append(row_idx)
                else:
                    transpose_idx_rel[col_idx] = [row_idx]
        
        #transpose_idx_vals =  [transpose_idx_rel[idx] for idx in range(len(transpose_idx_rel))]
        new = OneHotArray(shape=newshape, oha_dict=transpose_idx_rel)

        return new
    
    def __eq__(self, other):
        if isinstance(other, OneHotArray):
            return self.shape == other.shape and self.idx_rel == other.idx_rel
        elif isinstance(other, np.ndarray):
            if self.shape != other.shape:
                return False
            for i in range(other.shape[0]):
                for j in range(other.shape[1]):
                    val = other[i][j]
                    if val != 0 and val != 1:
                        return False
                    elif val == 1:
                        if j not in self.idx_rel[i]:
                            return False
                    elif val == 0:
                        if j in self.idx_rel[i]:
                            return False
                        
            return True
    
    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return str(self.idx_rel)

class RowSparseArray:
    """Row vectors assumed to be dense, column vectors assumed to be sparse, many zero vector rows"""

    def __init__(self, row_idx_vector, dense_row_array, total_array_rows, offset=0):
        """
        Args:
            row_idx_vector (numpy ndarray) : list of row indices that are dense
            dense_row_array (numpy ndarray) : stacked rows that are dense
            total_array_rows (tuple) : number of rows in the array
            offset (numeric, default = 0) : element-wise offset to true matrix
        
        """
        self._val_init(row_idx_vector, dense_row_array, total_array_rows, offset)

        self._row_idx_vector = row_idx_vector
        self._dense_row_array = dense_row_array
        self.shape = (total_array_rows, dense_row_array.shape[1])
        # offset for addition and subtraction
        self._offset = offset

    @staticmethod
    def _val_init(row_idx_vector, dense_row_array, total_array_rows, offset):
        
        if not isinstance(offset, (float, int)):
            raise TypeError("Offset must be int or float")

        if row_idx_vector.ndim != 1:
            raise Exception(f"Row index vector must be a vector with one dimension, not {row_idx_vector.ndim}")

        if dense_row_array.ndim != 2:
            raise Exception(f"Row index vector must be a vector with one dimension, not {dense_row_array.ndim}")

        if len(row_idx_vector) != len(dense_row_array):
            raise Exception(f"Every row index should correspond to exactly one row vector in the dense_row_array.\
                            There are {len(row_idx_vector)} row indices and {len(dense_row_array)} rows in dense row array")
        
        if np.unique(row_idx_vector).size != row_idx_vector.size:
            raise Exception("Cannot have duplicate row indices")
        
        if total_array_rows < len(row_idx_vector):
            raise Exception("Total array rows must be greater than or equal to dense row vectors")
    
    def to_array(self):
        
        array = np.full(self.shape, -self._offset)

        for row_idx, col_vals in zip(self._row_idx_vector, self._dense_row_array):
            array[row_idx] = col_vals - self._offset
 
        return array
    
    def subtract_from_update(self, array_to_update, copy=False):
        """Update numpy array by subtracting RowMatrixArray from numpy array to update
        
        Args:
            array_to_update (numpy ndarray) : array to update in memory
        """
        
        if self.shape != array_to_update.shape:
            raise Exception(f"Array shapes must be the size must be the same,\
                            RowSparseArray is {self.shape} while array to update is {array_to_update.shape}")

        if copy:
            array_to_update = np.copy(array_to_update)

        for row_idx, row_vec in zip(self._row_idx_vector, self._dense_row_array):
            array_to_update[row_idx] = array_to_update[row_idx] - row_vec

        if self._offset:
            array_to_update += self._offset
        
        return array_to_update
        

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if ufunc != np.add:
            raise Exception("Binary operation not supported")
        
        if ufunc == np.add:
            item1 = inputs[0]
            item2 = inputs[1]
            if isinstance(item2, RowSparseArray):
                return item2 + item1

    def __add__(self, other): 
        # have to support easy addition with numpy 
        if isinstance(other, (float, int)):
            new_offset = self._offset - other
            return RowSparseArray(row_idx_vector=self._row_idx_vector, 
                                  dense_row_array=self._dense_row_array, 
                                  total_array_rows=self.shape[0],
                                  offset=new_offset)
        if isinstance(other, np.ndarray):
            other = np.copy(other)
            for row_idx, row_vec in zip(self._row_idx_vector, self._dense_row_array):
                other[row_idx] = other[row_idx] + row_vec

            if self._offset:
                other -= self._offset
            return other
        return NotImplemented
    
    def __radd__(self, other):
        if isinstance(other, (float, int)):
            return self + other
        elif isinstance(other, np.ndarray):
            print("hello")
        else:
            return NotImplemented
    
    def __mul__ (self, other):
        if isinstance(other, (float, int)):
            return RowSparseArray(row_idx_vector=self._row_idx_vector, 
                                  dense_row_array=self._dense_row_array * other, 
                                  total_array_rows=self.shape[0],
                                  offset=self._offset * other)
        else:
            return NotImplemented
    
    def __rmul__(self,other):
        if isinstance(other, (float, int)):
            return self * other
        else:
            return NotImplemented
        
    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return self + -other
        else:
            return NotImplemented
        
    def __neg__(self):
        return -1 * self
        
    def __rsub__(self, other):
        if isinstance(other, (float, int)):
            return -1 * self + other
        else:
            return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, RowSparseArray):
            for attr, val in vars(self).items():
                try:
                    other_val = getattr(other, attr)
                except AttributeError:
                    return False
                
                if isinstance(val, np.ndarray):
                    if not np.allclose(val, other_val):
                        return False
                elif isinstance(val, (float, int)):
                    if not val == other_val:
                        return False
            return True
        elif isinstance(other, np.ndarray):
            pass
        else:
            return TypeError(f"Cannot evaluate equivalence between RowSparseMatrix and {type(other)}")

    def __str__(self):
        return f"Row Indices:\n{self._row_idx_vector}\nDense Array:\n{self._dense_row_array}\nShape: {self.shape}\nOffset: {self._offset}"

