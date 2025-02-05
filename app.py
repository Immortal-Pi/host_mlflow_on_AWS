import os 
import warnings
import sys 
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet 
from urllib.parse import urlparse
import mlflow 
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging 
from dotenv import load_dotenv
load_dotenv()

os.environ['MLFLOW_TRACKING_URI']=os.getenv('MLFLOW_TRACKING_URI')

logging.basicConfig(level=logging.WARN)
logger=logging.getLogger(__name__)

def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)
    return rmse,mae,r2 

if __name__=='__main__':

    ## Data ingestion - read the dataset 
    csv_url='https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv'
    try: 
        data=pd.read_csv(csv_url,delimiter=';')
        print(data)
    except Exception as e:
        logger.exception('unable to download the data')

    xtrain,xtest,ytrain,ytest=train_test_split(data.drop(columns=['quality']),data['quality'],test_size=0.2,random_state=42)

    alpha=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
    l1_ratio=float(sys.argv[2]) if len(sys.argv)>2 else 0.5 

    with mlflow.start_run():
        lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
        lr.fit(xtrain,ytrain)

        # prediction 
        predicted_quality=lr.predict(xtest)
        # print(predicted_quality)
        (rmse,mae,r2)=eval_metrics(predicted_quality,ytest)

        print(f'Elastic model (alpha={alpha}, l1_ratio:{l1_ratio})')
        print(f'RMSE:{rmse}')
        print(f'MAE:{mae}')
        print(f'R2:{r2}')

        mlflow.log_params({
            'alpha':alpha,
            'l1_ratio':l1_ratio,
        })
        mlflow.log_metrics({
            'rmse':rmse,
            'r2':r2,
            'mae':mae
        })

        ## for the remote server - setup AWS 
        remote_AWS_server_uri='http://ec2-44-201-168-70.compute-1.amazonaws.com:5000/'
        mlflow.set_tracking_uri(remote_AWS_server_uri)

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(lr,'model',registered_model_name='ElaseticNetWineModel')
        else:
            mlflow.sklearn.load_model(lr,'model')

    

