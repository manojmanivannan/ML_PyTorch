import numpy as np
import pandas as pd

# Function to create windowed sequences
def create_windows_feature_target(data_array, window_size, stride=1):
    """
    Returns a numpy array of windowed data with X as the features and y as the target
    Parameter:
        data_array: 2D numpy array or pandas dataframe
        window_size: int
        stride: int
    Return:
        X: 2D numpy array
        y: 1D numpy array
    """

    # check if data_array is an numpy array 
    if isinstance(data_array, np.ndarray):
        pass
  
    # check if data_array is a pandas dataframe
    elif isinstance(data_array, pd.DataFrame):
        data_array = data_array.to_numpy()
        
    else:
        raise ValueError("data_array must be a numpy array or pandas dataframe.")
    
    X, y = [], []
    for i in range(0, len(data_array) - window_size, stride):
        X.append(data_array[i:i + window_size])
        y.append(data_array[i + window_size])
    return np.array(X), np.array(y)
    
def get_mean_std_from_array(data_array):
    """
    A function to calculate the mean and standard deviation along axis 0 (vertical) of a numpy array or pandas dataframe.
    
    Parameters:
    data_array (numpy.ndarray or pandas.DataFrame): The input data to calculate mean and standard deviation from.
    
    Returns:
    tuple: A tuple containing the mean and standard deviation along axis 0 of the input data_array.
    """


    # check if data_array is an numpy array 
    if isinstance(data_array, np.ndarray):
        pass
  
    # check if data_array is a pandas dataframe
    elif isinstance(data_array, pd.DataFrame):
        data_array = data_array.to_numpy()
        
    else:
        raise ValueError("data_array must be a numpy array or pandas dataframe.")
    
    return np.mean(data_array, axis=0), np.std(data_array, axis=0)

def get_min_max_from_array(data_array):
    """
    A function to calculate the mean and standard deviation along axis 0 (vertical) of a numpy array or pandas dataframe.
    
    Parameters:
    data_array (numpy.ndarray or pandas.DataFrame): The input data to calculate mean and standard deviation from.
    
    Returns:
    tuple: A tuple containing the mean and standard deviation along axis 0 of the input data_array.
    """


    # check if data_array is an numpy array 
    if isinstance(data_array, np.ndarray):
        pass
  
    # check if data_array is a pandas dataframe
    elif isinstance(data_array, pd.DataFrame):
        data_array = data_array.to_numpy()
        
    else:
        raise ValueError("data_array must be a numpy array or pandas dataframe.")
    
    return np.min(data_array, axis=0), np.max(data_array, axis=0)

def normalize_mean_std(data_array, mean=None, std=None):
    """
    Normalize the data_array using the given mean and standard deviation.

    Args:
        data_array (numpy.ndarray or pd.DataFrame): The input data to be normalized.
        mean (float): The mean value to be used for normalization.
        std (float): The standard deviation value to be used for normalization.

    Returns:
        numpy.ndarray: The normalized data_array.
    """


    # check if data_array is an numpy array 
    if isinstance(data_array, np.ndarray):
        pass
  
    # check if data_array is a pandas dataframe
    elif isinstance(data_array, pd.DataFrame):
        data_array = data_array.to_numpy()
        
    else:
        raise ValueError("data_array must be a numpy array or pandas dataframe.")
    
    if mean is None or std is None:
        mean, std = get_mean_std_from_array(data_array)
    
    return (data_array - mean) / std

def normalize_min_max(data_array, min=None, max=None):
    """
    Normalize the data_array using the given mean and standard deviation.

    Args:
        data_array (numpy.ndarray or pd.DataFrame): The input data to be normalized.
        mean (float): The mean value to be used for normalization.
        std (float): The standard deviation value to be used for normalization.

    Returns:
        numpy.ndarray: The normalized data_array.
    """


    # check if data_array is an numpy array 
    if isinstance(data_array, np.ndarray):
        pass
  
    # check if data_array is a pandas dataframe
    elif isinstance(data_array, pd.DataFrame):
        data_array = data_array.to_numpy()
        
    else:
        raise ValueError("data_array must be a numpy array or pandas dataframe.")
    
    if min is None or max is None:
        min, max = get_min_max_from_array(data_array)
    
    return (data_array - min) / (max - min)

def train_test_split(features, target, split_ratio=0.8):
    """
    A function to split the data into train and test sets based on the given split ratio.

    Parameters:
    train_data (numpy.ndarray or pandas.DataFrame): The training data to be split.
    test_data (numpy.ndarray or pandas.DataFrame): The test data to be split.
    split_ratio (float, optional): The ratio of the data to be used for training. Default is 0.8.

    Returns:
    tuple: A tuple containing the training features, training target, validation features, validation target.
    """
    assert len(features) == len(target)

    split_index = int(len(features) * split_ratio)
    X_train, y_train = features[:split_index], target[:split_index]
    X_val, y_val = features[split_index:], target[split_index:]

    return X_train, y_train, X_val, y_val