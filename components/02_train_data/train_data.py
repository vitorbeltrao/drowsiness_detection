'''
Function that train custom images from your
labeled repository

Author: Vitor Abdo
Date: June/2023
'''

# import necessary packages
import logging
import timeit
import sys
import os
import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# config
DATA = sys.argv[1]
EPOCHS = int(sys.argv[2])
BATCH = int(sys.argv[3])
MODEL_NAME = sys.argv[4]
LR0 = int(sys.argv[5])
LRF = int(sys.argv[6])
WEIGHT_DECAY = int(sys.argv[7])


def train_custom_yolo_model(
    data: str,
    epochs: int,
    batch: int,
    model_name: str,
    lr0: float,
    lrf: float,
    weight_decay: float) -> tuple:
    '''
    Function to train a custom model using YOLO v8.

    Args:
    - data (str): Data configuration file path. Default is 'data.yaml'.
    - epochs (int): Number of training epochs. Default is 20.
    - batch (int): Batch size. Default is 8.
    - augment (bool): Flag to enable data augmentation. Default is True.
    - model_name (str): Output model name. Default is 'yolov8n_drowsiness'.
    - lr0 (float): Initial learning rate for optimizer. Default is 0.01.
    - lrf (float): Final learning rate for optimizer. Default is 0.01.
    - weight_decay (float): Weight decay for optimizer. Default is 0.0005.

    Returns:
    - results (Tuple): A tuple containing the training results.
    '''
    # Load the pretrained model
    model = YOLO('yolov8n.pt')

    # Training settings
    logging.info('Starting training...')
    logging.info(f'Data configuration: {data}')
    logging.info(f'Epochs: {epochs}')
    logging.info(f'Batch size: {batch}')
    logging.info(f'Model name: {model_name}')
    logging.info(f'Initial learning rate: {lr0}')
    logging.info(f'Final learning rate: {lrf}')
    logging.info(f'Weight decay: {weight_decay}')

    # Train the model
    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        name=model_name,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
    )

    logging.info('Training completed.')

    return results


if __name__ == "__main__":
    logging.info('About to start executing the train_data component\n')
    starttime = timeit.default_timer()

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    train_custom_yolo_model(DATA, EPOCHS, BATCH, MODEL_NAME, LR0, LRF, WEIGHT_DECAY)

    timing = timeit.default_timer() - starttime
    logging.info(f'The execution time of this step was:{timing}\n')
    logging.info('Done executing the train_data component')
