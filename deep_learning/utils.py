import gzip
import os
import traceback
import regex as re
from deep_learning import nn_layers

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
