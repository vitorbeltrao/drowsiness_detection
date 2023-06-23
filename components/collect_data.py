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

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


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


if __name__ == "__main__":
    # Create the directory structure if it doesn't exist
    data_dir = '../data'
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Execute the image collection
    labels = ['awake', 'drowsy']
    number_imgs = 20

    logging.info('About to start executing the image collect function')
    collect_images_from_webcam(labels, number_imgs, images_dir)
    logging.info('Done executing the image collect function')
