import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from flask import Flask, render_template, request, url_for


df = pd.read_csv("cleaned_data.csv",index_col=0)
df = df.drop('time',axis=1)

np.random.seed(1111)
raw_train, raw_test = train_test_split(df, test_size = 0.2)

X = df.drop(['viewCount'], axis = 1)
y = df['viewCount']

#split the data into training and test data
X_train, X_test, y_train,  y_test  = train_test_split(X, y, test_size = 0.2)

#RFE feature selection
model = LinearRegression()
rfe = RFE(estimator = model, n_features_to_select = 5)

rfe.fit(X_train, y_train)

selected_features = pd.DataFrame({'Feature': X_train.columns, 'Selected': rfe.support_, 'Rank': rfe.ranking_})


X_train = X_train[X_train.columns[rfe.support_]]
X_test = X_test[X_test.columns[rfe.support_]]

#XGBoost Model 
first_XBG = xgb.XGBClassifier()

first_XBG.fit(X_train, y_train)
first_prediction = first_XBG.predict(X_test)

xgb_accuracy_test = first_XBG.score(X_test, y_test)
xgb_accuracy_train = first_XBG.score(X_train, y_train)


print("Test score: ", xgb_accuracy_test)
print("Training score: ", xgb_accuracy_train)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(first_XBG, open('model.pkl','wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
sample = np.array([2,0,2017,1,14])
sample_reshaped = np.reshape(sample, (1, 5))
print(model.predict(sample_reshaped))
