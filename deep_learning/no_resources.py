import os
import tempfile
import numpy as np
import csv
import re
import time
import random
from contextlib import ExitStack
from . import utils

#TODO

## Merge OneHotTensor functionality with one hot array
    ## implement intelligent sizing in the one hot array

## option for generating ints or floats (maybe in set data input and output)

## chunk sizes mean batch sizes for chunk nn, but batch_size * train_prop = batch size for super chunk

## oha dict allows for double indexing (meaning can add 1s if given multiple of the same indexes)
## __iter__ and __next__ for generate
## apply one hot encoding for y_data just to OHA



# COMPLETED
## automatically create jar openers for setting input
## file extensions for jar should not come from name, but from actual file attributes  (see utils)


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

    opener = utils.JarOpener(source_path)
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

class OneHotTensor:
    """Multiple one hot arrays stacked in sequential order
    
    Creating so can comply with (num_examples, time_sequence, ...) standard shape of data factories
    and to increase efficiency in RNNs (like pad_packed_sequence) because no unnecessary 0 calculations will have to execute
    
    """

    def __init__(self, oha_list, uniform_shape_flag=True):
        """
        Args:
            uniform_shape (bool, default=True) : enforce shape set to False means arrays can have different shapes,
                shape of the tensor will be the largest dims"""
        self.ndim = 3

        # shape standard set by first oha that rest of ohas mus align to
        shape_standard = None
        self.uniform_shape_flag = uniform_shape_flag
        if self.uniform_shape_flag:
            for oha in oha_list:
                if not isinstance(oha, OneHotArray):
                    raise TypeError("list must be list of ohas")
                if shape_standard is None:
                    shape_standard = oha.shape
                else:
                    if oha.shape != shape_standard:
                        raise Exception("All one hot arrays must have the same shape")
        
            self.shape = (len(oha_list),) + shape_standard
        else:
            max_rows = 0
            max_cols = 0
            for oha in oha_list:
                num_rows, num_cols = oha.shape
                if num_rows > max_rows:
                    max_rows = num_rows
                if num_cols > max_cols:
                    max_cols = num_cols

            self.shape = (len(oha_list), max_rows, max_cols)
        self.oha_list = oha_list

    def __getitem__(self, key):
        """Defining getitem to duck type with numpy arrays for 0th axis slicing and indexing"""
        # define dimensions and n_rows placeholder
        
        # print(f"key {key}")
        # print(f"type key: {type(key)}")
        gathered = []
        if isinstance(key, int):
            self.add_int_key(key, gathered)
            if len(gathered) == 1:
                return gathered[0]
        elif isinstance(key, slice):
            self.add_slice_key(key, gathered)
            if len(gathered) == 1:
                return gathered[0]
        elif isinstance(key, np.ndarray):
            for elem in key:
                if isinstance(elem,tuple([int] + np.sctypes["int"])):
                    self.add_int_key(elem, gathered)
                else:
                    raise NotImplementedError
        elif isinstance(key, tuple):
            # gather the correct ohas and then perform getitem on them individually
            if isinstance(key[0], int):
                oha = self.oha_list[key[0]]
                return oha[key[1:]]
            else:
                temp_list = self.oha_list[key[0]]

            if isinstance(key[1], int):
                idx_rel = {}

                max_depth = 0
                for idx, oha in enumerate(temp_list):
                    if key[1] in oha.idx_rel.keys():
                        idx_rel[idx] = oha.idx_rel[key[1]]
                        if idx > max_depth:
                            max_depth=idx

                return OneHotArray(shape=(max_depth+1, self.shape[2]), oha_dict=idx_rel)[key[2]]
            else:
                temp_list = [oha[key[1]] for oha in temp_list]

            if isinstance(key[2], int):
                n_rows, n_cols = temp_list[0].shape[0],len(temp_list)
                idx_rel = {}
                for oha in temp_list:
                    for key, idx_list in oha.idx_rel.values():
                        idx_rel[key] = idx_rel.get(key, []) + idx_list

                return OneHotArray(shape=(n_rows, n_cols), oha_dict=idx_rel)
            else:
                gathered = [oha[key[2]] for oha in temp_list]

        else:
            raise SyntaxError
        
        return OneHotTensor(gathered, uniform_shape_flag=self.uniform_shape_flag)

    def add_int_key(self, int_idx, gathered):
        """Get integer index value 
        Args:
            int_idx (int) : integer row idx of the oha
            gathered (dict) : current gathered values of the indexed oha
        """
        self.validate_idx(int_idx)
        if int_idx < 0:
            int_idx = self.convert_neg_idx(int_idx)

        gathered.append(self.oha_list[int_idx])

    def convert_neg_idx(self, idx, axis=0):
        """Converts negative idxs for __getitem__ to positive idx
        Args:
            idx (int) : negative int to convert to positive
        """
        return self.shape[axis] + idx
    
    def validate_idx(self, idx, axis=0):
        """See if the idx is out of bounds or not
        
        Args:
            idx (int) :  index to validate
            axis (int, default=0)
        """
        indexed_rows = self.shape[axis]
        if idx < -indexed_rows or idx > indexed_rows-1:
            raise IndexError(f"Given index {idx} in axis {axis} does not exist")
        
    def add_slice_key(self, slice_obj, gathered):
        """Add corresponding valid index values in slice to gathered
        
        Args:
            slice (slice) : key slice object
            gathered (dict) : current gathered values of the indexed oha
        
        """
        start = 0 if slice_obj.start is None else slice_obj.start

        stop = self.shape[0] if slice_obj.stop is None else slice_obj.stop
        
        step = 1 if slice_obj.step is None else slice_obj.step
 
        for idx in range(start, stop, step):
            self.add_int_key(idx, gathered)

    def strip_flatten_to_array(self):
        """Convert to 2d array where there are no zero rows, one hot arrays are concatenated on top of each other
        Returns a large one hot array in regular array /large format"""

        num_filled_rows = 0

        for oha in self.oha_list:
            num_filled_rows += len(oha.idx_rel)

        strip_flattened_array = np.zeros((num_filled_rows, self.shape[-1]))

        row_num = 0
        for oha in self.oha_list:
            for row, one_locations in sorted(oha.idx_rel.items()): # these have to be sorted 
                for one_location in one_locations:
                    strip_flattened_array[row_num, one_location] = 1
                row_num+=1

        return strip_flattened_array
    
    def flip_ohas_copy(self):
        
        new_oha_list = []

        for oha in self.oha_list:
            new_oha_list.append(oha.flip_copy())

        return OneHotTensor(oha_list=new_oha_list, uniform_shape_flag=self.uniform_shape_flag)

    def __len__(self):
        return self.shape[0]
    
    def __str__(self):
        return str([str(oha) for oha in self.oha_list])

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

        # changing so can accommodate multiple dimensions
        if idx_array is not None and oha_dict == None:
            if type(idx_array) == np.ndarray: 
                if idx_array.ndim == 1:
                    idx_array = idx_array.reshape(1,-1)
            if self.shape[0] < len(idx_array):
                raise Exception("Number of row vectors in array must be greater than amount given")
            for row_idx, col_idxs in enumerate(idx_array):
                filtered_col_idxs = self.filter_col_idxs(col_idxs)
                if filtered_col_idxs:
                    cand_idx_rel[row_idx] = filtered_col_idxs

        elif oha_dict != None and idx_array is None:
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
            # both evaluate to false for np.nan's
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
                for row_idx in sorted(self.idx_rel.keys()):
                    col_idxs = self.idx_rel[row_idx]
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
        elif isinstance(key, (list, np.ndarray, tuple)):
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

        start = 0 if slice_obj.start is None else slice_obj.start

        stop = self.shape[0] if slice_obj.stop is None else slice_obj.stop
        
        step = 1 if slice_obj.step is None else slice_obj.step

        for idx in range(start, stop, step):
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
    
    def trim_tail(self):
        self.shape = (max(self.idx_rel)+1, self.shape[1])
        #self.shape[0] = max(self.idx_rel)

    def flip_copy(self):
        """Flip the one hot array about the 0th axis, returns a new array"""

        new_oha_dict = {}
        for row_num, one_locations in self.idx_rel.items():
            flipped_row = len(self) - row_num - 1 # -1 because zero indexed
            new_oha_dict[flipped_row] = one_locations

        return OneHotArray(shape=self.shape, oha_dict=new_oha_dict)


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
            raise NotImplementedError("Binary operation not supported")
        
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
        elif isinstance(other, np.ndarray):
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
            print("hello, other is an ndarray")
        else:
            return NotImplemented
    
    def __iadd__(self, other):
        if isinstance(other, (float, int)):
            self._offset -= other
        elif isinstance(other, RowSparseArray):
            assert self.shape == other.shape, "Operands must have the same shape"

            # print(f"self vector {self._row_idx_vector}")
            # print(f"other vector {other._row_idx_vector}")

            self_pointer = 0
            other_pointer = 0
            new_row_idx_vector = []
            new_dense_row_array_list = [] 

            while self_pointer < len(self._row_idx_vector) or other_pointer < len(other._row_idx_vector):

                self_row_num = self._row_idx_vector[self_pointer] if self_pointer != len(self._row_idx_vector) else np.inf
                other_row_num = other._row_idx_vector[other_pointer] if other_pointer != len(other._row_idx_vector) else np.inf

                if self_row_num < other_row_num:
                    new_row_idx_vector.append(self_row_num)
                    new_dense_row_array_list.append(self._dense_row_array[self_pointer])
                    self_pointer += 1
                elif self_row_num > other_row_num:
                    new_row_idx_vector.append(other._row_idx_vector[other_pointer])
                    new_dense_row_array_list.append(other._dense_row_array[other_pointer])
                    other_pointer += 1
                else: # they are equal
                    new_row_idx_vector.append(self_row_num)
                    new_dense_row_array_list.append(self._dense_row_array[self_pointer]+other._dense_row_array[other_pointer])
                    self_pointer += 1
                    other_pointer += 1
    
            self._row_idx_vector = np.array(new_row_idx_vector)
            self._dense_row_array = np.array(new_dense_row_array_list)

            # while self_pointer < len(self._row_idx_vector) and other_pointer < len(other._row_idx_vector):

            #     self_row_num = self._row_idx_vector[self_pointer]
            #     other_row_num = other._row_idx_vector[other_pointer]

            #     if self_row_num < other_row_num:
            #         new_row_idx_vector.append(self_row_num)
            #         new_dense_row_array_list.append(self._dense_row_array[self_pointer])
            #         self_pointer += 1
            #     elif self_row_num > other_row_num:
            #         new_row_idx_vector.append(other._row_idx_vector[other_pointer])
            #         new_dense_row_array_list.append(other._dense_row_array[other_pointer])
            #         other_pointer += 1
            #     else: # they are equal
            #         new_row_idx_vector.append(self_row_num)
            #         new_dense_row_array_list.append(self._dense_row_array[self_pointer]+other._dense_row_array[other_pointer])
            #         self_pointer += 1
            #         other_pointer += 1
        
            # if self_pointer == len(self._row_idx_vector):
            #     while other_pointer < len(other._row_idx_vector):
            #         new_row_idx_vector.append(other._row_idx_vector[other_pointer])
            #         new_dense_row_array_list.append(other._dense_row_array[other_pointer])
            #         other_pointer += 1

            # elif other_pointer == len(other._row_idx_vector):
            #     while self_pointer < len(self._row_idx_vector):
            #         new_row_idx_vector.append(self._row_idx_vector[self_pointer])
            #         new_dense_row_array_list.append(self._dense_row_array[self_pointer])
            #         self_pointer += 1

            # self._row_idx_vector = np.array(new_row_idx_vector)
            # self._dense_row_array = np.array(new_dense_row_array_list)
        else:
            return NotImplementedError
        
        return self

    
    def __mul__ (self, other):
        if isinstance(other, (float, int)):
            #print(self._row_idx_vector)
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

    def __len__(self):
        return self.shape[0]
