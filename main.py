'''
Main file to orchestrate the functions
made in components

Author: Vitor Abdo
Date: June/2023
'''

# import necessary packages
import logging
import playsound

from components.make_inferences import inference_video

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    #start here the code

    playsound.playsound('alarm.wav')