#!/usr/bin/env python3

"""
This is a sample program for the data science challenge and can be used as a
starting point for a solution.

It will be run as follows;
    sample_solution.py <current time> <input file name> <output file name>

Current time is the current hour and input file is all measured values from
the activation detector in each room for the past few hours.
"""
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import itertools
import joblib

# Reading input dataset and preparing it
def prepare_input_data(df):
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.ceil('H')
    #Keeping just one entry for each hour in a specific data for each device
    df.drop_duplicates(keep='first', inplace=True)
    # Aggregating per hour & filling (consider as inactive) missing hours for each device
    df = df.groupby('device'). \
        apply(lambda x: x.set_index('time').resample('H').mean().fillna(0)). \
        reset_index()
    df['hour'] = df['time'].dt.hour
    # Converting the device categorical data to numerical ones
    mlb = LabelEncoder()
    df['device'] = mlb.fit_transform(df['device'])
    return df

# Finding the best classifiers among nine sklearn classifier models for input dataset
def find_best_classifier(df):
    X = df[['device', 'hour']].values
    y = df['device_activated'].values
    # split input data to training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    acc = []
    model = []
    # Checking each classifier and saving the accuracy level of them
    for clf in classifiers:
        model.append(clf.fit(X_train, y_train))
        y_pred = clf.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred))
    indx = acc.index(max(acc))
    best_classifer = model[indx]
    # Saving the best classifiers as pickled model
    model_file_name = 'model/model.pkl'
    joblib.dump(best_classifer, model_file_name)
    return best_classifer

# Predict the occupacy of next 24 hours based the best found classifier
def prediction(previous_readings,predictions):
    best_classifier = find_best_classifier(previous_readings)
    predictions['hour'] = predictions['time'].dt.hour
    X = predictions[['device', 'hour']].values
    y_pred = best_classifier.predict(X)
    return y_pred

#Preparing the prediction output dataset
def predict_future_activation(current_time, previous_readings):
    # make predictable
    np.random.seed(len(previous_readings))
    # Make 24 predictions for each hour starting at the next full hour
    next_24_hours = pd.date_range(current_time, periods=24, freq='H').ceil('H')
    device_names = sorted(previous_readings.device.unique())

    # produce 24 hourly slots per device:
    xproduct = list(itertools.product(next_24_hours, device_names))
    predictions = pd.DataFrame(xproduct, columns=['time', 'device'])
    #predictions.set_index('time', inplace=True)
    predictions['activation_predicted'] = prediction(previous_readings,predictions)
    predictions.drop('hour', axis=1, inplace=True)
    # Converting the device numerical data to categorical ones
    predictions["device"].replace({0: "device_1", 1: "device_2", 2:"device_3",3:"device_4",4:"device_5",5:"device_6",6:"device_7"}, inplace=True)
    return predictions


if __name__ == '__main__':

    current_time, in_file, out_file = sys.argv[1:]

    df = pd.read_csv(in_file)
    previous_readings = prepare_input_data(df)
    result = predict_future_activation(current_time, previous_readings)
    result.to_csv(out_file)
