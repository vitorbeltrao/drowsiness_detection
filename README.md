# Drowsiness Detection with YOLOv8 - v0.0.1

## Table of Contents

1. [Project Description](#description)
2. [Files Description](#files)
3. [Running Files](#running)
4. [Inference and Alert System](#inference)
5. [Potential Applications](#applications)
6. [Licensing and Authors](#licensingandauthors)
***

## Project Description <a name="description"></a>

This project aims to detect drowsiness using [YOLOv8](https://docs.ultralytics.com/), a state-of-the-art object detection model. The goal is to create a custom model by training it on images collected from a webcam to detect signs of drowsiness, such as closed eyes or head drooping. Once drowsiness is detected, an audio alert is triggered to alert the person and prevent potential accidents. The project focuses on enhancing safety for individuals who drive long distances or work in industries where alertness is crucial, such as shift-based jobs.

![inferences](https://github.com/vitorbeltrao/drowsiness_detection/blob/main/images/drowsiness_detection_inferences.png?raw=true)
***

## Files Description <a name="files"></a>

* `data.yaml`: File that guides all model training. If in doubt, see the [documentation](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#1-create-dataset).

* `main.py file`: Main script in Python that runs all the components. All this managed by MLflow Projects.

* `components/`: Directory containing the modularized components for the project. The files listed here are more or less in the order they are called.

    * `collect_data.py`: Python module to collect images coming from your own webcam.
    * `train_data.py`: Python module to train your custom model using the own images you collected.
    * `make_inferences.py`: Python module to make inferences by a photo or video (real time).
    * `conda.yaml` and `MLproject` files: Mlflow scripts to manage all components above, respectively.

* `conda.yaml`: File that contains all the libraries and their respective versions so that the system works perfectly.

* `environment.yaml`: This file is for creating a virtual conda environment. It contains all the necessary libraries and their respective versions to be created in this virtual environment.

* `tests/`: directory that contains the tests for the functions that are in `components/`.

    * `test_collector.py`: Unit tests for the functions of the respective component (data_extract.py).
    * `test_transform.py`: Unit tests for the functions of the respective component (data_transform.py).
    * `test_load.py`: Unit tests for the functions of the respective component (data_load.py).
    * `conftest.py`: File where the fixtures were created to feed the unit tests.

* `alarm.wav`: File that contains the audio for the drowsiness detection.
***

## Running Files <a name="running"></a>
To run the project, follow these steps:

### Clone the repository

Go to [drowsiness_detection](https://github.com/vitorbeltrao/drowsiness_detection) and click on Fork in the upper right corner. This will create a fork in your Github account, i.e., a copy of the repository that is under your control. Now clone the repository locally so you can start working on it:

`git clone https://github.com/[your_github_username]/drowsiness_detection.git`

and go into the repository:

`cd drowsiness_detection` 

### Create the environment

Make sure to have conda installed and ready, then create a new environment using the *environment.yaml* file provided in the root of the repository and activate it. This file contain list of module needed to run the project:

`conda env create -f environment.yaml`
`conda activate drowsiness_detection`

### Get API key for Weights and Biases

Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to https://wandb.ai/authorize and click on the + icon (copy to clipboard), then paste your key into this command:

`wandb login [your API key]`

You should see a message similar to:

`wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc`

### 1° - Collect your own images

The first step in our pipeline is to get our images to feed the YOLOV8 custom model. To do this, just run `mlflow run . -P steps=collect_data`. 

With this command you will execute the first step of the pipeline and automatically your webcam will start, collect your images and move them to a folder in the main directory of the project called *data*. Inside the data folder it will automatically move the collected images randomly and proportionally dividing the images for *train*, *validation* and *test*.

After that you will have to label your images manually. For this we will use the [labelimg](https://github.com/heartexlabs/labelImg) package. After cloning the package here are the things that you need to do:

* Install `conda install -c anaconda pyqt` and `conda install -c conda-forge lxml` packages.

* Change to `cd labelimg` directory and run `python labelimg.py`. 

With that, the labelimg screen will open. To use labelimg, you should follow this [link](https://www.youtube.com/watch?v=tFNJGim3FXw&t=3139s) and and watch from minute *52:20*. This video, by this author, helped me a lot to develop this project!

### 2° - Train your custom model

After collecting the data, the second step is to train our custom model. To do this, just run `mlflow run . -P steps=train_data`. 

To train the model, we cannot forget to create the `data.yaml` file, to guide the model where it should find the data to be trained. At the end of the training, the final trained models will be in a folder called [bla] and you will be able to use them to make your inferences.

### 3° - Make inferences

After training the model in the previous step, you will use the best saved model to make inferences in real time on your webcam or on any image you want. To do this, just run `mlflow run . -P steps=make_inferences`. 

When running this command, your webcam will open automatically making inferences automatically.

### Testing

- Run the tests:

    `pytest`

    The tests of the functions used are in the `drowsiness_detection/tests` folder and to run them just write the code above in the terminal. In that folder are the tests that cover the production functions that are in the `drowsiness_detection/components` folder.
***

## Inference and Alert System <a name="inference"></a>

The `main.py` script performs real-time inference using the trained model on webcam data. It detects signs of drowsiness and triggers an audio alert if drowsiness is detected. You can customize the alert mechanism according to your requirements. The script uses the trained model weights generated during the training phase.
***

## Potential Applications <a name="applications"></a>

* Driver Safety: The drowsiness detection model can be integrated into vehicles to alert drivers when they exhibit signs of drowsiness, reducing the risk of accidents caused by fatigue.

* Shift-Based Jobs: The model can be deployed in industries where employees work in shifts, such as healthcare or transportation, to ensure that workers are alert and capable of performing their duties effectively.

* Personal Alert System: Individuals can use the model as a personal safety device to prevent drowsiness-related accidents during activities that require attentiveness, such as studying or operating machinery.
***

## Licensing and Author <a name="licensingandauthors"></a>

Vítor Beltrão - Data Scientist

Reach me at: 

- vitorbeltraoo@hotmail.com

- [linkedin](https://www.linkedin.com/in/v%C3%ADtor-beltr%C3%A3o-56a912178/)

- [github](https://github.com/vitorbeltrao)

- [medium](https://pandascouple.medium.com)

Licensing: [MIT LICENSE](https://github.com/vitorbeltrao/drowsiness_detection/blob/main/LICENSE)