# insurance-predictor using ANN and multiple linear regression
here we build a regressor model to predict the premium of a person

here we encodes the categorical variabkes and then apply multiple linear regression 

we get a root mean squared error of "5641"

we perform the same model with ann using keras and obtained a Rmse of "6041"

we aply 2 hidden layers with 8 nodes in the ANN model

ANN model is saved in "insurancepred.h5" file

By experiment with different models and different parameters we saw that random forest regression has the best performance over the model

By applyiing this we get a RMSE of "4376"

Random Forest model is saved in test.py file
