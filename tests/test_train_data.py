'''
Module to test function from
train_data component

Author: Vitor Abdo
Date: July/2023
'''

# import necessary packages
import pytest
from components.train_data import train_custom_yolo_model

def test_train_custom_yolo_model():
    '''Function to the function of train data component'''
    data = 'data.yaml'
    epochs = 3
    batch = 4
    model_name = 'test_yolov8'
    lr0 = 0.001
    lrf = 0.0001
    weight_decay = 0.0001

    results = train_custom_yolo_model(data, epochs, batch, model_name, 
                                      lr0, lrf, weight_decay)

    assert isinstance(results, tuple)
    assert len(results) > 0
