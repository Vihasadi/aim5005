# Importing MinMaxScaler, StandardScaler, LabelEncoder from aim5005
from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder  
import numpy as np
import unittest
from unittest.case import TestCase

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        # result not defined previously
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

# Reference links -https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
import unittest
import pytest

class TestLabelEncoder:
    
    def test_label_encoder_init(self):
        encoder = LabelEncoder()
        assert isinstance(encoder, LabelEncoder), "encoder is not a LabelEncoder object"
    
    def test_label_encoder_fit(self):
        encoder = LabelEncoder()
        labels = ['soccer', 'basketball', 'tennis', 'soccer']
        encoder.fit(labels)
        expected_mapping = {'basketball': 0, 'soccer': 1, 'tennis': 2}  # Sorted order
        assert encoder.label_index == expected_mapping, (
            f"Expected label index {expected_mapping}, got {encoder.label_index}"
        )
    
    def test_label_encoder_transform(self):
        encoder = LabelEncoder()
        labels = ['soccer', 'basketball', 'tennis']
        encoder.fit(labels)
        data = ['basketball', 'soccer', 'tennis']
        expected = [0, 1, 2]  # Based on the sorted mapping
        result = encoder.transform(data)
        assert (np.array(result) == np.array(expected)).all(), (
            f"Expected transformed values {expected}, got {result}"
        )
    
    def test_label_encoder_fit_transform(self):
        encoder = LabelEncoder()
        labels = ['soccer', 'basketball', 'tennis']
        expected = [1, 0, 2]  # Based on the sorted order
        result = encoder.fit_transform(labels)
        assert (np.array(result) == np.array(expected)).all(), (
            f"Expected transformed values {expected}, got {result}"
        )
    
    def test_label_encoder_single_value(self):
        encoder = LabelEncoder()
        labels = ['soccer', 'basketball', 'tennis']
        encoder.fit(labels)

        # Transforming a single value
        single_value = ['soccer']
        expected = [1]  # Based on the sorted mapping
        result = encoder.transform(single_value)

        assert (np.array(result) == np.array(expected)).all(), (
            f"Expected transformed value {expected}, got {result}"
        )
