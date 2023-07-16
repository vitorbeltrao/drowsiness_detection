'''
Function that test the trained model
in test set

Author: Vitor Abdo
Date: July/2023
'''

# import necessary packages
import logging
import sys
import wandb
# import cv2
# import matplotlib.pyplot as plt
from ultralytics import YOLO
# from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# config
YOLO_MODEL_PATH = sys.argv[1]
IMAGE_PATH = sys.argv[2]


# def inference_image(image_path: str, yolo_model_path: str) -> None:
#     '''
#     Perform inference on an image.

#     Args:
#         image_path (str): The path to the input image.
#         yolo_model_path (str): The path to the YOLO model weights.

#     Returns:
#         None
#     '''
#     # Load the trained model
#     final_model = YOLO(yolo_model_path)
#     logging.info("Loading image: %s", image_path)

#     # Load the image
#     img = Image.open(image_path)
#     logging.info("Performing inference on the image...")

#     # Perform inference
#     results = final_model(img)
#     logging.info("Inference completed. Displaying results.")

#     # Plot and display the results
#     res_plotted = results[0].plot()
#     recolor = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
#     plt.imshow(recolor)
#     plt.show()


if __name__ == "__main__":
    logging.info('About to start executing the test data component\n')
    # inference_image(YOLO_MODEL_PATH, IMAGE_PATH)
    # save the results in wandb
    run = wandb.init(
        project='drowsiness_detection',
        entity='vitorabdo',
        job_type='Yolo test model')
    run.finish()
    
    logging.info('Creating run for drowsiness detection: SUCCESS\n')
    logging.info('Done executing the test data component')
