'''
Function that make real time
inferences with video

Author: Vitor Abdo
Date: June/2023
'''

# import necessary packages
import logging
import sys
import cv2
import pygame
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# config
YOLO_MODEL_PATH = sys.argv[1]
ID_VIDEO = sys.argv[2]


def inference_video(yolo_model_path: str, id_video: int) -> None:
    '''
    Perform real-time inference using the webcam.

    Args:
        yolo_model_path (str): The path to the YOLO model weights.
        id_video (int): id of the video capturing device to open.

    Returns:
        None
    '''
    # Load the trained model
    final_model = YOLO(yolo_model_path)
    logging.info("Connecting to the webcam...")

    # Connect to the webcam
    cap = cv2.VideoCapture(id_video)

    # Loop through each frame until we close the webcam
    while cap.isOpened():
        ret, frame = cap.read()
        logging.info("Performing inference on the current frame...")

        # Perform inference on the current frame
        results = final_model(frame)
        annotated_frame = results[0].plot()

        # Display the frame with annotations
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        logging.info("Press 'q' to stop the inference.")

        # Check if the 'q' key is pressed and break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()

    # Close the frame window
    cv2.destroyAllWindows()


def play_alarm_sound():
    # Inicialize o mixer do pygame
    pygame.mixer.init()

    # Carregue o arquivo de som
    sound = pygame.mixer.Sound('alarm.wav')

    # Reproduza o som do alarme
    sound.play()

    # Aguarde até que o som termine de ser reproduzido
    pygame.time.wait(int(sound.get_length() * 1000))

    # Pare de reproduzir o som e libere os recursos
    sound.stop()
    pygame.mixer.quit()


if __name__ == "__main__":
    logging.info('About to start executing the real time inferences component\n')
    inference_video(YOLO_MODEL_PATH, ID_VIDEO)
    logging.info('Done executing the real time inferences component')
