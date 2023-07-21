'''
This is the main system file that runs all necessary
components to run the machine learning pipeline

Author: Vitor Abdo
Date: June/2023
'''

# import necessary packages
import argparse
import os
import mlflow

# define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=str, default='all', help='Steps to execute')

_steps = [
    # 'collect_data',
    'train_data',
    'test_data',
    'deployment'
]


def main():
    '''Main file that runs the entire pipeline end-to-end using mlflow
    :param steps: str
    Steps to execute. Default is 'all', which executes all steps
    '''
    # read command line arguments
    args = parser.parse_args()

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ['WANDB_PROJECT'] = 'drowsiness_detection'
    os.environ['WANDB_RUN_GROUP'] = 'development'

    # Steps to execute
    steps_par = args.steps
    active_steps = steps_par.split(',') if steps_par != 'all' else _steps

    if 'collect_data' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/drowsiness_detection#components/collect_data'
        mlflow.run(project_uri, parameters={'steps': 'collect_data'})

    if 'train_data' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/drowsiness_detection#components/train_data'
        mlflow.run(project_uri, parameters={'steps': 'train_data'})

    if 'test_data' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/drowsiness_detection#components/test_data'
        mlflow.run(project_uri, parameters={'steps': 'test_data'})
    
    if 'deployment' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/drowsiness_detection#components/deployment'
        mlflow.run(project_uri, parameters={'steps': 'deployment'})


if __name__ == "__main__":
    # call the main function
    main()
