# Azure Nanodegree Capstone Project: Breast Cancer Prediction

This is my capstone project for the Udacity Azure ML Engineer Nanodegree.

The objective is to train a machine learning model using Hyperdrive and AutoML and compare the results. Models from these two experiments are compared using Accuracy score, the best model is registered and deployed to Azure Container Service as a REST endpoint with key based authentication.

## Dataset

### Overview
I have chosen to use a dataset that i found on kaggle (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). THis is a dataset of Breast Cancer Wisconsin (Diagnostic) Dataset. 
Breast Cancer is the most common found in women. It starts when cells in the breast begin to grow out of control. These cells usually form tumors which can be seen in an X-ray or felt as lumps in the breast area.

About the Dataset:
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The distribution of Diagnosis Class is made up of 357 benign and 212 malignant.

Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

### Task
The main challenge for detection is classifying tumors into malignant (cancerous) or benign (non cancerous). As such the object of this machine learning project is a binary classification, where we will use machine learning to predict the Diagnosis as either as Malignant (M) or Benign (B). I will be utilizing AutoML and Hyperdrive to find which has the best precition accuracy.

### Access
The compressed dataset is available to download in Kaggle and i was able to add to my Github so that i can register it as a dataset in my Azure workspace in the Azure ML Studio GUI.

## Automated ML
For the automated ML run, I setup the following:

<img width="404" alt="automlconfigsettings" src="https://github.com/samfrost23/nd00333-capstone/assets/99268262/6c6cfeaa-be8f-46af-929a-5c74c2e22523">


As the target problem is to predict a diagnosis in the form of an M or a B this is a binary prediction and so the task type should be classification on the Diagnosis label. Limiting the experiment to a total duration of 1 hr to reduce the potential for session timeout. The primary metric that I used as an objective for the autoML run was weighted AUC, which is the area under the curve and it is the suggested metric for anomaly detection in the Azure documentation. 

AutoML Settings:
experiment_timeout_minutes - This defines how long experement will run in mins
max_concurrent_iterations - The maximum number of iterations that would be executed in parallel.
n_cross_validations - Number of cross validations to perform
primary_metric - The metric that Automated Machine Learning will optimize for model selection

AutoML Config:
I have set the following for the AutoML Config
enable_early_stopping - Early termination if the score is not improving in the short term
enable_onnx_compatible_models - to enable Onnx-compatitble models
label_column_name - The column that will be predicted


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
Here are the models trained by Automated ML:

![readme automl](https://github.com/samfrost23/nd00333-capstone/assets/99268262/1d7a8e8c-86d1-45af-9c00-3ff5de9a6b39)


![readme automl results](https://github.com/samfrost23/nd00333-capstone/assets/99268262/166ec05c-9042-4053-8fec-ffbf473285cd)


The best results were:
![automl_bestrun](https://github.com/samfrost23/nd00333-capstone/assets/99268262/67235231-67f7-411f-b739-2904d0a759b1)


details of some of the metrics of the best model:

![readme automl best model](https://github.com/samfrost23/nd00333-capstone/assets/99268262/94d35966-abcd-4920-beab-cff9677ac0ad)


I ran the RunDetails widget to follow the progress

<img width="429" alt="RunDetails" src="https://github.com/samfrost23/nd00333-capstone/assets/99268262/2108872c-f2cd-46f0-94a0-96ead65fe240">
<img width="449" alt="RunDetails1" src="https://github.com/samfrost23/nd00333-capstone/assets/99268262/79d4de5e-085c-4030-a985-ef64188df23b">

<img width="325" alt="RunDetails2" src="https://github.com/samfrost23/nd00333-capstone/assets/99268262/f78006e1-2509-43a1-95f9-d9e063ae468e">


The metrics can be shown using get_metrics() 

![automl_bestrun](https://github.com/samfrost23/nd00333-capstone/assets/99268262/1988cbed-4ed5-4426-b8f0-036a0a362baa)

I saved the model and also converted the model to ONNX format

![onnx1](https://github.com/samfrost23/nd00333-capstone/assets/99268262/3af0e6e7-f245-4081-8e12-5ad4262ff197)

![onnx](https://github.com/samfrost23/nd00333-capstone/assets/99268262/487fb59a-fd1a-480a-8e24-677f37864d74)


I deployed the model as a webservice

![automl webservicesetup](https://github.com/samfrost23/nd00333-capstone/assets/99268262/8a885ad5-f5a4-4c48-9341-162d84b2abb3)


Checked the health of the service

![automl check health](https://github.com/samfrost23/nd00333-capstone/assets/99268262/f9cd641e-9987-4ef0-a474-f38fa959a0ef)


I then created a test using some test data to make sure it is working
![automl test](https://github.com/samfrost23/nd00333-capstone/assets/99268262/4727306f-9ef2-4359-8e27-a3823b3fabc9)

I also enabled app insights as this helps for troubleshooting
![automl app insights](https://github.com/samfrost23/nd00333-capstone/assets/99268262/20d29c08-8741-4846-827f-732f13edb9b2)

![automl app insights2](https://github.com/samfrost23/nd00333-capstone/assets/99268262/1cb47417-77ac-4f13-81a5-c4d66ef85189)

![automl app insights3](https://github.com/samfrost23/nd00333-capstone/assets/99268262/64c0860a-55aa-455b-8e74-86133f34f77e)


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
To compare against AutoML I used the Scikit-learn Logistic Regression as a classification algorithm.

I used the RandomForestClassifier model for this experiment, this adds randomness to the model and searches for the best feature.

The HyperDriveConfig class was configured with the following parameter sampler:

![hd ps](https://github.com/samfrost23/nd00333-capstone/assets/99268262/98658607-0737-4f54-9cd5-42c4a4e944af)



--n_estimators: The number of trees within the forest
--min_samples: Minimum Number of samples
--max_features: Number of features to use

With the config set to:

![hd config](https://github.com/samfrost23/nd00333-capstone/assets/99268262/7e744e18-de12-4090-a3fc-3ecf5505da9c)


Primary_metric_name as Accuracy and primary_metric_goal was set to maximize.

The Accuracy was calculated on the test set for each run, best model was then retrieved and saved

I chose RandomParameterSampling because it supports early termination of low performance runs which helps to reduce computation time and still allows us to to find reasonably good models, this saves us time and cost for computing resources.

![early term](https://github.com/samfrost23/nd00333-capstone/assets/99268262/33834b4f-81ca-44fa-9bdc-45fe74b13b0f)


I used BanditPolicy for this reason which is an "aggressive" early stopping policy. With BanditPolicy it defines an early termination policy based on a slack factor and evaluation interval which is specified as 0.1, 2 respectively:

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
Here are the models trained by Automated ML:



The best results were:

details of some of the metrics of the best model:


I ran the RunDetails widget to follow the progress


The metrics can be shown using get_metrics() 
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
