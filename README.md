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


Limiting the experiment to a total duration of 1 hr to reduce the potential for session timeout. The primary metric that I used as an objective for the autoML run was weighted AUC, which is the area under the curve and i is the suggested metric for anomaly detection in the Azure documentation. 

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

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
