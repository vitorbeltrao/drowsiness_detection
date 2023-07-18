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
                 final_model: str,
                 latest_scores_path: str,
                 ingested_files_path: str) -> None:
    '''Function for deployment, copy the latest pickle file, the latestscore.txt value, 
    and the ingestedfiles.txt file into the deployment directory

    :param prod_deployment_path: (str)
    Folder in the main directory where the production files will be copied

    :param final_model: (pickle)
    Pickle file with all saved model pipeline

    :param latest_scores_path: (str)
    Directory of the "06_test_model" folder where the latest model score file is located

    :param ingested_files_path: (str)
    Directory of the "01_upload_raw_data" folder where the "ingested_files" is located
    '''
    logging.info(
        f'Model deployment. Model, its latest score, and the list of files used for training are saved in '
        f'{prod_deployment_path}')

    # obtaining the necessary files
    # 1. final model pickle
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='deployment')
    model_local_path = run.use_artifact(final_model, type='pt').download()
 
    # create production deployment folder if it doesnt exists
    if not os.path.isdir(prod_deployment_path):
        os.mkdir(prod_deployment_path)

    # copy files to the production deployment folder
    shutil.copy(os.path.join(model_local_path, 'best_model.pt'), prod_deployment_path) # copy pickle
    logging.info('Copied files: SUCCESS')


if __name__ == '__main__':
    logging.info('About to start executing the deployment function')
    deploy_model(PROD_DEPLOYMENT_PATH, FINAL_MODEL)
    logging.info('Done executing the deployment function')