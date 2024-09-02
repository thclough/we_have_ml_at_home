import gzip
import os
import traceback
import regex as re
from deep_learning import nn_layers, no_resources
import numpy as np

def pos_int(cand, var_name):
    """Checks if the passed value is a positive integer
    
    Args:
        cand (float) : value to check if positive integer or not
        var_name (str) : name of variable to use in errors raised
    """
    if type(cand) != int:
        raise TypeError(f"{var_name} must be an integer")
        
    if cand < 1: 
        raise TypeError(f"{var_name} must be positive")       

def get_file_dir(source_path: str) -> str:
    """Retrieve directory of source path file"""

    pattern = r'(\S+/{1})\S+'

    # Use re.search to find the pattern in the file path
    match = re.search(pattern, source_path)
    
    # If a match is found, return the matched file extension
    if match:
        return match.group(1)
    else:
        return None  # Return None if no extension is found
    
def dim_size(dims):
    if isinstance(dims, tuple):
        size = dims[0]
    elif isinstance(dims, int):
        size = dims
    else:
        raise TypeError("Dims must be int or tuple")
    
    return size

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

# GRAPH UTILS

def find_graph_start_nodes(graph_dict):
    """Find the starting nodes of a graph represented by a dictionary
    
    Args:
        graph_dict (dict) : {input_nodes : list_of_neighbor_nodes}

    Returns:
        starting_nodes (list) : list of nodes with no incoming edge
    """

    incoming_count_dict = {}
    
    {key:0 for key in graph_dict.keys()}

    for node in graph_dict:
        # add node to dict if not already in dict
        incoming_count_dict[node] = incoming_count_dict.get(node, 0)
        for neighbor in graph_dict[node]:
            incoming_count_dict[neighbor] = incoming_count_dict.get(neighbor, 0) + 1

    starting_nodes = [node for node in incoming_count_dict if incoming_count_dict[node]==0]

    return starting_nodes
        
def reverse_graph(graph_dict):
    """Reverses the edges of a digraph and returns an appropriate dictionary of the new graph
    
    Args:
        graph_dict (dict) : {input_nodes : list_of_neighbor_nodes}

    Returns:
        reversed_graph_dict (dict) : {input_nodes : list_of_neighbor_nodes} (of reversed graph)
    """
    
    reversed_graph_dict = {}

    for start_node, end_nodes in graph_dict.items():
        for end_node in end_nodes:
            value = reversed_graph_dict.get(end_node, [])
            value.append(start_node)
            reversed_graph_dict[end_node] = value

    return reversed_graph_dict

def graph_dict_to_edge_list(graph_dict):
    """create list of edges from graph dict
    
    Args:
        graph_dict (dict) : {input_nodes : list_of_neighbor_nodes}

    Returns:
        edge_list (list) : list of connections
    """
    edge_list = []

    for start_node, end_nodes in graph_dict.items():
        for end_node in end_nodes:
            edge_list.append((start_node, end_node))

    return edge_list

def one_hot_labels(y_data, one_hot_width):
    """One hot labels from y_data

    Args:
        y_data (numpy array)
        one_hot_width (int) : number of one hot categories

    Returns:
        one_hot_labels

    """

    one_hot_labels = np.zeros((y_data.size, one_hot_width))
    one_hot_labels[np.arange(y_data.size), y_data.astype(int).flatten()] = 1

    return one_hot_labels

def adding_with_padding(first_val, second_val):
    """adding two dimensional arrays, first value to second value in place of first value. 
    1st dimension must be the same, but if 0th dimension is different, will pad the shorter with zeroes
    to add to the longer


    Args:
        first_val (array-like)
        second_val (array-like)
    """
    
    # handle integers
    first_shape_len = len(first_val)
    second_shape_len = len(second_val)

    if first_shape_len == second_shape_len:
        first_val += second_val
    elif first_shape_len > second_shape_len:
        second_val = np.pad(second_val, ((0, first_shape_len-second_shape_len), (0,0)), mode="constant")
        np.add(first_val, second_val, out=first_val)
    elif first_shape_len < second_shape_len:
        first_val = np.pad(second_val, ((0, second_shape_len-first_shape_len), (0,0)), mode="constant")
        np.add(first_val, second_val, out=first_val)
    
def accordion(array, desired_len):
    """Shorten or stretch (pad) len of array to desired length
    
    Args:
        array (np.ndarray) : array to modify
        desired_len (int) : 

    Returns:
        array of desired length
    """

    array_len = len(array)

    if desired_len < array_len:
        return array[:desired_len]
    elif desired_len > array_len:
        pad_list = [(0, desired_len-array_len)] + [(0,0)] * (array.ndim - 1)
        return np.pad(array, tuple(pad_list), mode="constant")
    else:
        return array

def zero_lpad(array_to_pad, pad_width):
    """pads 2d array to the left with zeroes by the pad width specified
    
    Args:
        array_to_pad (numpy.ndarray) : array that needs left padding
        pad_width (int) : number of zeroes to the left
    """
    return np.pad(array_to_pad, ((0, 0), (pad_width,0)), mode="constant", constant_values=0)

def lol_flatten(list_of_lists):
    """Flatten a list of lists, by combining lists in list, 

    Args:
        list_of_lists (list)

    Returns
        combined_list (list) : combined lists into one list
    
    """

    combined_list = []

    for member_list in list_of_lists:
        combined_list += member_list

    return combined_list

def flip_the_lists_in_the_list(list_of_lists):
    return [item[::-1] for item in list_of_lists]

def lol_flip_flatten(list_of_lists):
    
    return lol_flatten(flip_the_lists_in_the_list(list_of_lists))

def flip_time(batch_outputs):
    """Flip timesteps axis on (num_examples, timesteps, categories), 
    used for backwards jointed model"""

    if type(batch_outputs) == np.ndarray:
        return np.flip(batch_outputs, 1)
    elif type(batch_outputs) == no_resources.OneHotTensor:
        return batch_outputs.flip_ohas_copy()
    elif type(batch_outputs) == list:
        return lol_flip_flatten(batch_outputs)
    else:
        raise NotImplementedError

def flatten_batch_outputs(batch_outputs):
    """Flattens the outputs of a model into an array with dimensions (total_num_outputs_over_time, 1)
    
    Args:
        batch_outputs (array-like) : outputs, possibly separated into multiple timesteps

    Returns:
        batch_outputs_flattened (np.ndarray) : (total_num_outputs_over_time, 1) array
    """

    if type(batch_outputs) == np.ndarray:
        if batch_outputs.ndim > 2:
            return batch_outputs.reshape(-1, batch_outputs.shape[-1])
        else:
            return batch_outputs
    elif type(batch_outputs) == no_resources.OneHotTensor:
        return batch_outputs.strip_flatten_to_array()
    elif type(batch_outputs) == list:
        return np.array(lol_flatten(batch_outputs))
    else:
        raise NotImplementedError
    
def array_list_product(array_list):
    """Find product of list of numpy arrays of same dimensions

    Args:
        array_list (list) : list of numpy arrays to use as factors
        
    Returns:
        product of the arrays
        
    """

    return np.prod(np.stack(array_list, axis=0), axis=0)


def join_paths(starts_of_paths, ends_of_paths):
    """Joins path sections in starts_of_paths to path sections in ends_of_paths for paths of layers
    
    Args:
        starts_of_paths (list of lists of 2-tuples of layers) : first parts of paths
        ends_of_paths (list of lists of 2-tuple of layers) : second parts of paths

    Returns:
        connected_paths (list of lists of 2-tuples of layers) : final paths each with one part from first 
    """

    connected_paths = []

    if len(starts_of_paths) != len(ends_of_paths):
        raise Exception("Number of starts must be equal to number of ends")

    for start, end in zip(sorted(starts_of_paths, key=lambda x : x[-1][-1].str_id), sorted(ends_of_paths, key=lambda x : x[0][0].str_id)):
        if start[-1][-1] != end[0][0]:
            raise Exception("Start paths and end paths do not match up, ")
        connected_paths.append(start+end)
    
    return connected_paths
         
