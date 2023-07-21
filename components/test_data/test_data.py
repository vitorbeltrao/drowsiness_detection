'''
Function that test the trained model
in test set

Author: Vitor Abdo
Date: July/2023
'''

# import necessary packages
import logging
import sys
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# config
YOLO_MODEL_PATH = sys.argv[1]
TEST_DATA_PATH = sys.argv[2]


def inference_image(image_path: str, yolo_model_path: str) -> None:
    '''
    Perform inference on an image.

    Args:
        image_path (str): The path to the input image.
        yolo_model_path (str): The path to the YOLO model weights.

    Returns:
        None
    '''
    # Load the trained model
    final_model = YOLO(yolo_model_path)
    logging.info("Loading image: %s", image_path)

    # Load the image
    img = Image.open(image_path)
    logging.info("Performing inference on the image...")

    # Perform inference
    results = final_model(img)
    logging.info("Inference completed. Displaying results.")

    # Plot and display the results
    res_plotted = results[0].plot()
    recolor = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    plt.imshow(recolor)
    plt.show()


if __name__ == "__main__":
    logging.info('About to start executing the test data component\n')

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    filenames = os.listdir(TEST_DATA_PATH)
    for image in filenames:
        inference_image(TEST_DATA_PATH + image, YOLO_MODEL_PATH)

    logging.info('Done executing the test data component')
