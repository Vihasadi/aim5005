import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        return (x-self.minimum)/diff_max_min
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    import numpy as np
    def __init__(self):
        self.mean = None
        self.stdev = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. 
        If it can't be cast raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a np.ndarray"
        return x
        
    def fit(self, x: np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.stdev = x.std(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize the given vector.
        """
        x = self._check_is_array(x)
        # Use the input x instead of self.x
        diff_mean = x - self.mean
        return diff_mean / self.stdev
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

'''Label encoder used external resource- https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/,
    https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values'''
class LabelEncoder:
    def __init__(self):
        self.label_index = {}
        self.index_label = {}
## converting to sets to remove duplicates and also model learns about the labels
    def fit(self, labels):
        unique_labels = sorted(set(labels))  # Sort labels for consistent mapping
        for index, label in enumerate(unique_labels):
            self.label_index[label] = index
            self.index_label[index] = label
##  Converting original labels to indices
    def transform(self, labels):
        return [self.label_index[label] for label in labels]

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)
