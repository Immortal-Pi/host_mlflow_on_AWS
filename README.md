# MLFLOW on AWS 

how to host mlflow on AWS EC2 instance

# MLflow on AWS setup:

1. Login to AWS console 
2. Create IAM user with Administrator access 
3. Export the credentials in your AWS CLI by running 'aws configure'
4. create a s3 bucket 
5. create EC2 machine (Ubuntu) & add security group 5000 port 

# Run the following commands on EC2 machine 

```bash

sudo apt update 
sudo apt install python3-pip
sudo apt install pipenv 
sudo apt install virtualenv 
mkdir mlflow 
cd mlflow 
pipenv install mlflow 
pipenv install awscli
pipenv install boto3
pipenv shell 

```
## Then set AWS credentials on the EC2 instance 
aws configure

# Finally 
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflowtrackingwine

##  open Public IPv4 DNS to the port 5000

##  set uri in your local terminal and in your code 

example url 
```bash
export MLFLOW_TRACKING_URI=http://ec2-44-201-168-70.compute-1.amazonaws.com:5000/
```
or set the environment variables in .env file 

## run the application 
the arguments are alpha and l1_ratio values for ElasticNet 

```bash
python app.py 0.1 0.5
```
# screenshots 

![Mlflow1](https://github.com/Immortal-Pi/host_mlflow_on_AWS/blob/main/outputs/1.png)

![Mlflow2](https://github.com/Immortal-Pi/host_mlflow_on_AWS/blob/main/outputs/2.png)

![AMAZONS3](https://github.com/Immortal-Pi/host_mlflow_on_AWS/blob/main/outputs/3.png)
