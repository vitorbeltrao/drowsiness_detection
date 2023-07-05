'''
Function that collects images from your webcam

Author: Vitor Abdo
Date: June/2023
'''

# import necessary packages
import os
import cv2
import time
import uuid
import logging
import sys
import random
import shutil
import wandb

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# config
DATA_DIR = sys.argv[1]
NUMBER_IMGS = int(sys.argv[2])
TRAIN_RATIO = float(sys.argv[3])
VAL_RATIO = float(sys.argv[4])
TEST_RATIO = float(sys.argv[5])


def collect_images_from_webcam(
        labels: list, number_imgs: int, images_path: str) -> None:
    '''
    Collects images from the webcam and saves them to a folder.

    Args:
        labels (list): List of labels for the images.
        number_imgs (int): Number of images to be collected for each label.
        images_path (str): Path to the folder where the images will be saved.

    Returns:
        None
    '''
    cap = cv2.VideoCapture(0)

    # Loop through labels
    for label in labels:
        print('Collecting images for {}'.format(label))
        time.sleep(5)

        # Loop through image range
        for img_num in range(number_imgs):
            logging.info(f'Collecting images for {label}, image number {img_num}')

            # Webcam feed
            ret, frame = cap.read()

            # Naming our image path
            imgname = os.path.join(images_path, label + '.' + str(uuid.uuid1()) + '.jpg')

            # Write out image to file
            cv2.imwrite(imgname, frame)

            # Render to the screen
            cv2.imshow('Image Collection', frame)

            # 3-second delay between captures
            time.sleep(3)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                logging.info(f'You broke the sequence of collection for {label} images')
                break

    cap.release()
    cv2.destroyAllWindows()

def split_images(images_path: str, labels_path: str):
    '''
    Split the collected images into train, val, and test sets.

    Args:
        images_path (str): Path to the folder where the images are stored.
        labels_path (str): Path to the folder where the labels are stored.

    Returns:
        None
    '''
    dest_folders = ['train', 'val', 'test']
    for folder in dest_folders:
        os.makedirs(os.path.join(images_path, folder), exist_ok=True)
        os.makedirs(os.path.join(labels_path, folder), exist_ok=True)

    file_list = os.listdir(images_path)
    random.shuffle(file_list)

    num_files = len(file_list)
    train_count = int(TRAIN_RATIO * num_files)
    val_count = int(VAL_RATIO * num_files)

    for i, file_name in enumerate(file_list):
        if file_name.endswith('.jpg'):
            src_path = os.path.join(images_path, file_name)

            if i < train_count:
                dest_folder = 'train'
            elif i < train_count + val_count:
                dest_folder = 'val'
            else:
                dest_folder = 'test'
            
            dest_path = os.path.join(images_path, dest_folder)
            shutil.move(src_path, dest_path)

            logging.info(f'Moved {file_name} to {dest_folder} folder.')


if __name__ == "__main__":
    logging.info('About to start executing the collect_data component\n')
    
    # Create the directory structure if it doesn't exist
    images_dir = os.path.join(DATA_DIR, 'images')
    labels_dir = os.path.join(DATA_DIR, 'labels')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Execute the image collection
    labels = ['awake', 'drowsy']

    logging.info('About to start executing the image collect function')
    collect_images_from_webcam(labels, NUMBER_IMGS, images_dir)
    logging.info('Done executing the image collect function\n')

    # Split the images into train, val, and test sets
    split_images(images_dir, labels_dir)

    run = wandb.init()
    artifact = wandb.Artifact('drowsiness', type='dataset')
    artifact.add_dir(DATA_DIR + '/')
    run.log_artifact(artifact) 
    logging.info('Uploaded dataset to wandb: SUCCESS\n')

    logging.info('Done executing the collect_data component')
