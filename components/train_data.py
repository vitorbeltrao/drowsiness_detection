'''
Function that train custom images from your
labeled repository

Author: Vitor Abdo
Date: June/2023
'''

# import necessary packages
import logging
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def train_custom_yolo_model(
    data: str = '../data.yaml',
    epochs: int = 20,
    batch: int = 8,
    model_name: str = 'yolov8n_drowsiness',
    lr0: float = 0.01,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    device: str = None) -> tuple:
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
    - device (int or str): device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu. Default is None.

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
    logging.info(f'Device: {device}')

    # Train the model
    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        name=model_name,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        device=device
    )

    logging.info('Training completed.')

    return results
