---
title: 'Build, train, and deploy, a machine learning model with Amazon SageMaker notebook instance'
date: 2021-10-10
permalink: /posts/2021/08/aws-sagemaker/
tags:
  - xgboost
  - classification
  - amazon 
  - amazon sagemaker
  - sagemaker studio
  - sagemaker notebook instance
  - Build, train, and deploy, a machine learning model
  - imbalanced classification
  - binary classification
---


#  **Why amazon SageMaker?**

Amazon SageMaker helps Machine Learning (ML) Engineers, Data Scientists and developers to prepare, build, train, and deploy ML models quickly. 


<img src="https://d1.awsstatic.com/SageMaker/SageMaker%20reInvent%202020/most_comprehensive_box_no_space%402x.5e35d9542b9311059942552d3804241c9621bf77.png">


 Amazon SageMaker supports the leading machine learning frameworks, toolkits and programming languages like jupyter notebook, TensorFlow, PyTorch, mxNet, etc.


# **Build, train, and deploy, a machine learning model with Amazon SageMaker notebook instance**

An Amazon SageMaker notebook instance is a machine learning compute instance running the Jupyter Notebook App. Use Jupyter notebooks in your notebook instance to prepare and process data, write code to train models, deploy models to SageMaker hosting, and test or validate your models. 



To build, train and deploy the ML model, we should have the following resources:

* IDE/ Jupyter-notebook
* Data stored in someplace
* ML Algorithm
* Place to store trained model

In this section, we will use the Amazon SageMaker Notebook instance to build, train, and deploy an XGBoost model.

*To use the Amazon SageMaker Notebook instance, you should have an AWS account.*

* *If you don't have an account go to https://portal.aws.amazon.com/billing/signup to create an account. After you have created the account, the dashboard looks like the following image.*

<img src = "https://drive.google.com/uc?export=view&id=11LyaJOzgDRKRhgoQTZ06bM_X6O-aiQCL" >

* *If you already have an AWS account go to https://aws.amazon.com/console/ and sign in with your credentials. The dashboard looks like the above image.*



# How to start an Amazon SageMaker notebook instance?

1. To create the SageMaker notebook instance, click on search and type *"sagemaker"* as shown in the following figure.

<img src = "https://drive.google.com/uc?export=view&id=1Ny-DCXsID3VDWOzMi63von1VOGNsCF7K" >

2. Click on amazon SageMaker.

<img src = "https://drive.google.com/uc?export=view&id=1mlhErTVb5Aa2RDV-x223HDOqFL6X0g3M" >

3. After clicking on amazon SageMaker, the following window will appear. Click on notebook and then notebook instance. 
<img src = "https://drive.google.com/uc?export=view&id=121KcUGy9VgCQz_5qwGvBr1cZDbC5RyiC" >

4. To create a notebook instance click on create notebook instance as shown in the following figure.

<img src = "https://drive.google.com/uc?export=view&id=1-aAZqcg0waJzx-vfB366BwSkMoQSkEeN" >

5. After clicking the create notebook instance button, the following window appears. Fill in the necessary details as shown in the figure.
    * Add the Notebook instance name. For our example, we will name it *"MyProject"*.

<img src = "https://drive.google.com/uc?export=view&id=1xEsmPjd00Tx6nI7jZiHLtrNMiPGVsc2b" >

   * Scroll down to change the Identity and Access Management(IAM) role. To learn more about IAM, visit https://aws.amazon.com/iam/faqs/.
    
<img src = "https://drive.google.com/uc?export=view&id=1ulmEjxuACHURzSLGGRuc3X_ln4dC8Cxn" >


   * Select *"create a new role"*.
   
<img src = "https://drive.google.com/uc?export=view&id=1rSNCdAWHfRvJV-F5JrSxv8PnTFi63B6K" >


   * After clicking *"create a new role"* the following pop-up window appears. We will provide access to any S3 bucket. However, you can change the configuration as you need. Finally, click on Create role button.
   
<img src = "https://drive.google.com/uc?export=view&id=152P5qst8kTRBBdfHQe_1GzOyb-Tp1k0b" >

   * The IAM role is successfully created, as shown in the following diagram.
   
<img src = "https://drive.google.com/uc?export=view&id=1-o4a0Oyw0QswT5hwJA2q5lHh0K4xk97L" >

  * Everything is set now. Let's scroll down and create the notebook instance.
  
<img src = "https://drive.google.com/uc?export=view&id=1WO_1Lctswb8lpaImYkcFzhjxFd_GQtaM" >


After creating the notebook instance, the dashboard looks like the following image. It will take a couple of minutes to display the status in service from the pending state. When everything is green, click on *"open jupyter"* to open jupyter notebook in the AWS SageMaker cloud. 

<img src = "https://drive.google.com/uc?export=view&id=1Irr_Qrtob_ZspSGTV8UxHyWZ83t71SiX" >



The jupyter notebook opened looks precisely like the python jupyter notebook in the local machine. 
<img src = "https://drive.google.com/uc?export=view&id=1ErjICJmSpEat36E0E2b21EKaRwq5AlUa" >
We can create a new notebook by clicking a new button and then select "*conda_python3*" because we will be using python3  and a built-in XgBoost model. You can choose other variations as per your need.
<img src = "https://drive.google.com/uc?export=view&id=1q1X9S7bs0iXkcQRTi-XvsGrXlflo0tqp" >


The jupyter-notebook is created.

<img src = "https://drive.google.com/uc?export=view&id=1D3C_p-AZtwmOkkLiSGH12gzysZFpsizf" >



We have successfully created Jupyter-notebook. Now it is time to download the data and store it in s3 bucket. We are storing the dataset in S3 to maintain proper versioning. 

S3 is also used to store models. Therefore the essential steps in this pipeline will be 


* Importing necessary Libraries
* Creating an S3 bucket
* Mapping train And Test Data in S3
* Mapping The path of the models in S3

# Import libraries


```python
import sagemaker 
import boto3
from sagemaker.session import Session 
```

# Create S3 bucket

Check the region


```python
my_region =  boto3.session.Session().region_name # set the region of the instance
print(my_region)
```

    us-east-2


Orovide the bucket name


```python
bucket_name = 'bapp-test-buck'
```

Create s3 resource using boto3


```python
s3 = boto3.resource('s3')
```

Create a bucket if the region is *us-east-2*. The region might be different on your machine. Please check the region name using `boto3.session.Session().region_name` command.


```python
try:
    if my_region == 'us-east-2':
        # create bucket:
        s3.create_bucket(Bucket = bucket_name, CreateBucketConfiguration={'LocationConstraint': 'us-east-2'} )
except Exception as e:
    print(str(e))
```

S3 Bucket is created. To look into the bucket goto dashboard and search S3 and click the S3 tiles.
<img src = "https://drive.google.com/uc?export=view&id=14M3gzheWxmQPrT0WtL4hdLJEXiygIRJh" >


# Download dataset and store it in s3

The entire dataset is downloaded form the [given link.](https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv).


```python
import pandas as pd
import urllib
try:
    urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ',e)
```

    Success: downloaded bank_clean.csv.


Let's read the downloaded dataset.


```python
try:
    model_data = pd.read_csv('./bank_clean.csv',index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)
```

    Success: Data loaded into dataframe.


Split the dataset into a train and test set. After splitting data into train and test sets, they are stored in the S3 bucket on the train and test folder, respectively. 


```python
### Train Test split
import numpy as np
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)
```

    (28831, 61) (12357, 61)


Before uploading the train and test dataset, make sure the dependent variable is in the first column for the XgBoost algorithm. 

# Create a pipeline or path to store model in s3 bucket


```python
# set an output path where the trained model will be saved
prefix = 'xgboost-as-a-built-in-algo'
output_path ='s3://{}/{}/output'.format(bucket_name, prefix)
print(output_path)
```

    s3://bapp-test-buck/xgboost-as-a-built-in-algo/output



```python
### Saving Train And Test Into Buckets
## We start with Train Data
import os
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], 
                                                axis=1)], 
                                                axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')

```

The dataset is stored in s3 bucket `baap-test-buck/xgboost-as-a-built-in-algo/train/` as shown in the following figure.

<img src = "https://drive.google.com/uc?export=view&id=1rS_eHYRK2_7HkURyNj3gVihGuZhWd0m_" >

Find the url of train data.


```python
s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')
s3_input_train
```




    <sagemaker.inputs.TrainingInput at 0x7f4758f01128>



Similarly for the test data.


```python
pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')

```

Find the url of test data.


```python
s3_input_test = sagemaker.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')
s3_input_test
```




    <sagemaker.inputs.TrainingInput at 0x7f4758644080>



<img src = "https://drive.google.com/uc?export=view&id=1EFB7kb2OpXAtpzd0JUR85VNmz13cgCC8" >

# Download the xgboost algorithm image container of version 1.0-1


```python
from sagemaker.amazon.amazon_estimator import image_uris

container =  image_uris.retrieve(region=boto3.Session().region_name, framework='xgboost', version='1.0-1')
```

Initialize the hyperparameters for the XgBoost algorithm. Here the hyperparameters are engineered to form GridSearchCV or RandomizedSearchCV algorithm. The steps to derive hyperparameters is not shown in this notebook because computing them takes time and resources. Moreover, doing everything in the cloud makes no sense because it costs money. 


```python
# initialize hyperparameters
hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"binary:logistic",
        "num_round":50
        }

```

Construct a SageMaker estimator that calls the xgboost-container to handle end-to-end Amazon SageMaker training and deployment tasks.


To construct the SageMaker estimator, specify the following parameters:


* image_uri – Specify the training container image URI. In this example, the SageMaker XGBoost training container URI is specified using SageMaker.image_uris.retrieve.

* role – The AWS Identity and Access Management (IAM) role that SageMaker uses to perform tasks on your behalf (for example, reading training results, call model artefacts from Amazon S3, and writing training results to Amazon S3).

* instance_count and instance_type – The type and number of Amazon EC2 ML compute instances for model training. For this training exercise, you use a single ml.m4.xlarge instance, 4 CPUs, 16 GB of memory, an Amazon Elastic Block Store (Amazon EBS) storage, and high network performance. For more information about EC2 compute instance types, see Amazon EC2 Instance Types

* volume_size – The size, in GB, of the EBS storage volume to attach to the training instance. If you use File mode, this must be large enough to store training data (File mode is on by default).

* output_path – The path to the S3 bucket where SageMaker stores the model artefact and training results.

* use_spot_instances (bool) – Specifies whether to use SageMaker Managed Spot instances for training. If enabled, then the max_wait arg should also be set. [More Information](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html) (default: False).
    
* max_wait (int) – Timeout in seconds waiting for spot training job (default: None). After this amount of time, Amazon SageMaker will stop waiting for managed spot training job to complete (default: None).




```python
estimator = sagemaker.estimator.Estimator(image_uri=container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          instance_count=1, 
                                          instance_type='ml.m5.2xlarge', 
                                          volume_size=5, # 5 GB 
                                          output_path=output_path,
                                          use_spot_instances=True,
                                          max_run=300,
                                          max_wait=600)


```

Launch training.


```python
estimator.fit({'train': s3_input_train,'validation': s3_input_test})
```

    2021-10-09 18:42:56 Starting - Starting the training job...
    2021-10-09 18:43:19 Starting - Launching requested ML instancesProfilerReport-1633804976: InProgress
    ...
    2021-10-09 18:43:50 Starting - Preparing the instances for training............
    2021-10-09 18:45:56 Downloading - Downloading input data
    2021-10-09 18:45:56 Training - Training image download completed. Training in progress..[34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
    [34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value binary:logistic to Json.[0m
    [34mReturning the value itself[0m
    [34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
    [34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[18:45:58] 28831x59 matrix with 1701029 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[18:45:58] 12357x59 matrix with 729063 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Single node training.[0m
    [34mINFO:root:Train matrix has 28831 rows[0m
    [34mINFO:root:Validation matrix has 12357 rows[0m
    [34m[18:45:58] WARNING: /workspace/src/learner.cc:328: [0m
    [34mParameters: { num_round } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    [0m
    [34m[0]#011train-error:0.10079#011validation-error:0.10528[0m
    [34m[1]#011train-error:0.09968#011validation-error:0.10456[0m
    [34m[2]#011train-error:0.10017#011validation-error:0.10375[0m
    [34m[3]#011train-error:0.09989#011validation-error:0.10310[0m
    [34m[4]#011train-error:0.09996#011validation-error:0.10286[0m
    [34m[5]#011train-error:0.09906#011validation-error:0.10261[0m
    [34m[6]#011train-error:0.09930#011validation-error:0.10286[0m
    [34m[7]#011train-error:0.09951#011validation-error:0.10261[0m
    [34m[8]#011train-error:0.09920#011validation-error:0.10286[0m
    [34m[9]#011train-error:0.09871#011validation-error:0.10294[0m
    [34m[10]#011train-error:0.09868#011validation-error:0.10294[0m
    [34m[11]#011train-error:0.09868#011validation-error:0.10326[0m
    [34m[12]#011train-error:0.09854#011validation-error:0.10358[0m
    [34m[13]#011train-error:0.09892#011validation-error:0.10342[0m
    [34m[14]#011train-error:0.09850#011validation-error:0.10342[0m
    [34m[15]#011train-error:0.09844#011validation-error:0.10326[0m
    [34m[16]#011train-error:0.09857#011validation-error:0.10318[0m
    [34m[17]#011train-error:0.09799#011validation-error:0.10318[0m
    [34m[18]#011train-error:0.09816#011validation-error:0.10383[0m
    [34m[19]#011train-error:0.09857#011validation-error:0.10383[0m
    [34m[20]#011train-error:0.09830#011validation-error:0.10350[0m
    [34m[21]#011train-error:0.09826#011validation-error:0.10318[0m
    [34m[22]#011train-error:0.09847#011validation-error:0.10399[0m
    [34m[23]#011train-error:0.09833#011validation-error:0.10407[0m
    [34m[24]#011train-error:0.09812#011validation-error:0.10415[0m
    [34m[25]#011train-error:0.09812#011validation-error:0.10399[0m
    [34m[26]#011train-error:0.09774#011validation-error:0.10375[0m
    [34m[27]#011train-error:0.09781#011validation-error:0.10375[0m
    [34m[28]#011train-error:0.09781#011validation-error:0.10391[0m
    [34m[29]#011train-error:0.09778#011validation-error:0.10367[0m
    [34m[30]#011train-error:0.09781#011validation-error:0.10383[0m
    [34m[31]#011train-error:0.09771#011validation-error:0.10358[0m
    [34m[32]#011train-error:0.09743#011validation-error:0.10391[0m
    [34m[33]#011train-error:0.09753#011validation-error:0.10342[0m
    [34m[34]#011train-error:0.09767#011validation-error:0.10342[0m
    [34m[35]#011train-error:0.09757#011validation-error:0.10350[0m
    [34m[36]#011train-error:0.09757#011validation-error:0.10342[0m
    [34m[37]#011train-error:0.09736#011validation-error:0.10342[0m
    [34m[38]#011train-error:0.09750#011validation-error:0.10342[0m
    [34m[39]#011train-error:0.09733#011validation-error:0.10350[0m
    [34m[40]#011train-error:0.09705#011validation-error:0.10358[0m
    [34m[41]#011train-error:0.09701#011validation-error:0.10383[0m
    [34m[42]#011train-error:0.09712#011validation-error:0.10407[0m
    [34m[43]#011train-error:0.09698#011validation-error:0.10375[0m
    [34m[44]#011train-error:0.09733#011validation-error:0.10342[0m
    [34m[45]#011train-error:0.09736#011validation-error:0.10367[0m
    [34m[46]#011train-error:0.09746#011validation-error:0.10350[0m
    [34m[47]#011train-error:0.09736#011validation-error:0.10358[0m
    [34m[48]#011train-error:0.09712#011validation-error:0.10334[0m
    [34m[49]#011train-error:0.09712#011validation-error:0.10318[0m
    
    2021-10-09 18:46:19 Uploading - Uploading generated training model
    2021-10-09 18:46:19 Completed - Training job completed
    Training seconds: 34
    Billable seconds: 7
    Managed Spot Training savings: 79.4%


<img src = "https://drive.google.com/uc?export=view&id=1h3ebyU7Xli5g9-wkDbLIEeTvZ447hZJ-" >

# Deploy model As Endpoints

We can deploy the model using the `deploy()` API. The created endpoint can be seen on the Endpoints section of the SageMaker console and in the Endpoints tab of SageMaker Studio. 


```python
xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
```

    -------------!

# Get Test data


```python
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.serializer = sagemaker.serializers.CSVSerializer() # set the serializer type
```

# Predict test data using deployed xgb_predictor.

We can use the `predict()` API 
to send it a CSV sample for prediction.


```python
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
```


```python
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)
```

    (12357,)



```python
predictions_array
```




    array([0.05214286, 0.05660191, 0.05096195, ..., 0.03436061, 0.02942475,
           0.03715819])



# Confusion matrix for classification


```python
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
```

    
    Overall Classification Rate: 89.7%
    
    Predicted      No Purchase    Purchase
    Observed
    No Purchase    91% (10785)    34% (151)
    Purchase        9% (1124)     66% (297) 
    


# Delete the endpoints and every resources if you don't need them


When we're done working with the endpoint, we shouldn't forget to delete it to avoid unnecessary charges. Deleting an endpoint is as simple as calling the `delete_endpoint()` API: 


```python
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint_name)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()
```




    [{'ResponseMetadata': {'RequestId': 'VJ48T5NW5QM21MRE',
       'HostId': 'bpnpgK4S+N4+NpXpB6SERzAV1TdPEI0BwHDequ4g+a2aQygtTNLgdIdPGhK0YrQtGUm8WkKqeDU=',
       'HTTPStatusCode': 200,
       'HTTPHeaders': {'x-amz-id-2': 'bpnpgK4S+N4+NpXpB6SERzAV1TdPEI0BwHDequ4g+a2aQygtTNLgdIdPGhK0YrQtGUm8WkKqeDU=',
        'x-amz-request-id': 'VJ48T5NW5QM21MRE',
        'date': 'Sat, 09 Oct 2021 18:58:27 GMT',
        'content-type': 'application/xml',
        'transfer-encoding': 'chunked',
        'server': 'AmazonS3',
        'connection': 'close'},
       'RetryAttempts': 0},
      'Deleted': [{'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/BatchSize.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/OverallFrameworkMetrics.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/profiler-output/system/incremental/2021100918/1633805100.algo-1.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-report.ipynb'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/profiler-output/system/incremental/2021100918/1633805160.algo-1.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/LowGPUUtilization.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/OverallSystemUsage.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/profiler-output/framework/training_job_end.ts'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/CPUBottleneck.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/output/model.tar.gz'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/GPUMemoryIncrease.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/IOBottleneck.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/MaxInitializationTime.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/profiler-output/system/training_job_end.ts'},
       {'Key': 'xgboost-as-a-built-in-algo/test/test.csv'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-report.html'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/StepOutlier.json'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/LoadBalancing.json'},
       {'Key': 'xgboost-as-a-built-in-algo/train/train.csv'},
       {'Key': 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-10-09-18-42-56-149/rule-output/ProfilerReport-1633804976/profiler-output/profiler-reports/Dataloader.json'}]}]



# Credits

[1] Code Credit: [Krish Naik Youtube channel](https://www.youtube.com/watch?v=LkR3GNDB0HI)

[2] https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html

[3] https://aws.amazon.com/sagemaker/

[4] https://aws.amazon.com/sagemaker/studio/



```python

```
