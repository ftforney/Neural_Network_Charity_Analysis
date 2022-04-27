# Neural Network Charity Analysis

## Purpose

We will use deep-learning neutral networks by utilizing TensorFlow in Python. Our goal is to analyze and class the success of donations.

## Results

### Pre-Processing

 - Our target column is the `IS_SUCCESSFUL` column, saying if a donation was used effectively or not.
 - Our feature columns are `APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT`. These will need to be encoded so our model can interpret the data from these features.
 - `EIN` and `NAME` columns hold no value to training our model or predicting data, so they will be removed.

### Compiling, Training, and Evaluation

 - In this model, we have two hidden layers made up of 80 and 30 neurons in order. We will be using `ReLU` as the activation function for our hidden layers since it is generally inexpensive, and has good performance. For output, we will use the `sigmoid` function for its probability output.
 - The accuracy of this model never reached 75% or above, even with an attempt on optimization.
 - For attempted optimization, we tried bucketing the feature `ASK_AMT`. For a second attempt, we tried increasing the number of neurons on a hidden layer, and in addition tried to add another hidden layer. As a last effort, we tried using the `tanh` activation function.

## Summary

In the end, we were not able to meet our 75% goal, but we have a model that we can now build off of and alter in ways to try to inch closer to that goal as time goes on. It might be best to use a different method of machine learning here, such as Supervised Learning with the Random Forest Classifier. Using trees may be able to increase our accuracy, and we can compare that with the model we have built here.
