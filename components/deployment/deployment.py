'''
Function that will copy the files in 
"prod_deployment_path" to organize them

Author: Vitor Abdo
Date: July/2023
'''

# Import necessary packages
import os
import logging
import sys
import wandb
import shutil

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')

# config
PROD_DEPLOYMENT_PATH = sys.argv[1]
FINAL_MODEL = sys.argv[2]


def deploy_model(prod_deployment_path: str, 
                 final_model: str) -> None:
    '''Function for deployment, copy the latest pickle file, the latestscore.txt value, 
    and the ingestedfiles.txt file into the deployment directory

    :param prod_deployment_path: (str)
    Folder in the main directory where the production files will be copied

    :param final_model: (pickle)
    Yolo weights file with all saved model pipeline
    '''
    logging.info(
        f'Model deployment. Lets save the best trained model in '
        f'{prod_deployment_path}')

    # obtaining the necessary files
    run = wandb.init(
        project='drowsiness_detection',
        entity='vitorabdo',
        job_type='deployment')
    model_local_path = run.use_artifact(final_model, type='pt').download()
    run.finish()
 
    # create production deployment folder if it doesnt exists
    if not os.path.isdir(prod_deployment_path):
        os.mkdir(prod_deployment_path)

    # copy files to the production deployment folder
    shutil.copy(os.path.join(model_local_path, 'best_model_pipe.pt'), prod_deployment_path) # copy .pt file
    logging.info('Copied files: SUCCESS')


if __name__ == '__main__':
    logging.info('About to start executing the deployment function')
    deploy_model(PROD_DEPLOYMENT_PATH, FINAL_MODEL)
    logging.info('Done executing the deployment function')
