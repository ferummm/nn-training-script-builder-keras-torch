import pandas as pd
import numpy as np
import os

def load_csv_data(filename, output_cols, drop_na = False, col_names = False, inp_shape = None, out_shape = None, reshape_single_output = False):
    """
    Loads data from a CSV file and returns input and output arrays.
    """
    if not filename.endswith('.csv'):
            raise ValueError('Input file must be a CSV file')
    if not type(inp_shape) == type(out_shape) == tuple:
        raise ValueError("Must provide shape parameters as a tuple")
    if not isinstance(output_cols, list) and not all(isinstance(elem, int) for elem in output_cols):
        raise ValueError("Must provide output columns as a list of integers")
    try:
        df = pd.read_csv(filename) if col_names else pd.read_csv(filename, header=None)
        df = df.dropna() if drop_na else df
        data = df[1:] if col_names else df
        input_cols = [x for x in range(df.shape[1]) if x not in output_cols]

        input_data = data.iloc[:, input_cols]
        output_data = data.iloc[:, output_cols]
        
        input_data = np.array(input_data).reshape((-1,) + inp_shape[1:])
        output_data = np.array(output_data).reshape((-1,) + out_shape[1:])
        
        assert inp_shape[1:] == input_data.shape[1:], "Input shape does not match input data shape"
        assert out_shape[1:] == output_data.shape[1:], "Output shape does not match output data shape"
        assert input_data.shape[0] == output_data.shape[0], "Input and output data have different batch sizes"
        if reshape_single_output:
            if (len(out_shape) == 2 and out_shape[1] == 1):
                output_data = output_data.reshape(-1,)
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except pd.errors.ParserError:
        raise ValueError("CSV file could not be parsed")
    except FileNotFoundError:
        raise ValueError("CSV file not found")
    except:
        raise ValueError("Data from file must agree with the shape parameters")
    return np.array(input_data, dtype = 'float32'), np.array(output_data,dtype = 'float32')

def load_excel_data(filename, output_cols, drop_na = False, col_names = False, inp_shape = None, out_shape = None, reshape_single_output = False):
    """
    Loads data from a excel file and returns input and output arrays.
    """
    file_ext = os.path.basename(filename).split('.')[1].lower()
    if not file_ext in ('xlx', 'xlsx'):
        raise ValueError('Input file must be a excel file')
    if not type(inp_shape) == type(out_shape) == tuple:
        raise ValueError("Must provide shape parameters as a tuple")
    if not isinstance(output_cols, list) and not all(isinstance(elem, int) for elem in output_cols):
        raise ValueError("Must provide output columns as a list of integers")
    try:
        df = pd.read_excel(filename) if col_names else pd.read_excel(filename, header=None)
        df = df.dropna() if drop_na else df
        data = df[1:] if col_names else df
        input_cols = [x for x in range(df.shape[1]) if x not in output_cols]

        input_data = data.iloc[:, input_cols]
        output_data = data.iloc[:, output_cols]

        input_data = np.array(input_data).reshape((-1,) + inp_shape[1:])
        output_data = np.array(output_data).reshape((-1,) + out_shape[1:])
        
        assert inp_shape[1:] == input_data.shape[1:], "Input shape does not match input data shape"
        assert out_shape[1:] == output_data.shape[1:], "Output shape does not match output data shape"
        assert input_data.shape[0] == output_data.shape[0], "Input and output data have different batch sizes"
        if reshape_single_output:
            if (len(out_shape) == 2 and out_shape[1] == 1):
                output_data = output_data.reshape(-1,)
            """  input_data = np.array(input_data).reshape((-1,) + inp_shape[1:])
        output_data = np.array(output_data) """
    except pd.errors.EmptyDataError:
        raise ValueError("Excel file is empty")
    except FileNotFoundError:
        raise ValueError("Excel file not found")
    except IndexError:
        raise ValueError("Data from file must agree with the output indexes parameters")
    except:
        raise ValueError("Data from file must agree with the shape parameters")
    return np.array(input_data, dtype = 'float32'), np.array(output_data,dtype = 'float32')

def load_data(filename, res_cols, drop_na = False, col_names = False, inp_shape = None, out_shape = None, reshape_single_output=False):
    """
    Loads data from a CSV or XLSX file and returns input and output arrays.

    Args:
    filename (str): path to the CSV file.
    res_cols (list): list of integers representing the column indexes of the output data.
    drop_na (bool): whether to drop rows with NaN values (default: False).
    col_names (bool): whether the CSV file has column names (default: False).
    inp_shape (tuple): shape of the input data (default: None).
    out_shape (tuple): shape of the output data (default: None).

    Returns:
    input_data (numpy.ndarray): float32 array of input data.
    output_data (numpy.ndarray): float32 array of output data.
    """
    file_ext = os.path.basename(filename).split('.')[1].lower()
    if file_ext in ('csv'):
        return load_csv_data(filename, res_cols, drop_na, col_names, inp_shape, out_shape,reshape_single_output)
    elif file_ext in ('xlsx','xlx'):
        return load_excel_data(filename, res_cols, drop_na, col_names, inp_shape, out_shape,reshape_single_output)
    else:
        raise ValueError("Invalid file format, use excel or csv files")
    
