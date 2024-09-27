#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:40:18 2024

@author: anastassiakustenmacher

Notes after preparing a feature analyse with Univariate Feature Selection, RFE and Random Forest
We can conclude that statistical features such as _skewness, _kurtosis, _entropy are not relevant
"""
import importlib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

from sklearn.metrics import classification_report, accuracy_score

import utils
from utils import load_df, plot_label
# Reload the utils module to reflect changes
importlib.reload(utils)

single_file = True # Load data from a single file

df=load_df(single_file)
# Check the distribution of the target variable (activity labels)
print(Counter(df['label']))  # Assuming 'activity' is the column with class labels

# Initialize a list to store features for each subsequence
features_list = []
labels = []

accel_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']

# Handle missing values (if any)
df = df.dropna()
window = 50
step=25

# Iterate over the data frame grouped by activity and extract features
for label, group in df.groupby('label'):
    # Reset the index of the group
    group = group.reset_index(drop=True)
    
    # Split the group into subsequences of window-length samples
    #for start in range(0, len(group), window):
    for start in range(0, len(group), step):
        #end = min(start + window, len(group))
        end = min(start + step, len(group))
        subsequence = group.iloc[start:end]
        
        # Ensure subsequence is from the same activity
        if len(subsequence['label'].unique()) == 1:
            # Extract frequency features from the subsequence
            time_features = utils.extract_statistical_features(subsequence, accel_columns)
            features_list.append(time_features)
            #subsequence_features = utils.extract_features(subsequence, accel_columns)
            #features_list.append(subsequence_features)
            labels.append(label)

# Convert the list of features into a data frame
features_df = pd.DataFrame(features_list)
labels = np.array(labels)
features_df['Label'] = labels
X = features_df.drop(columns=['Label'])  # Features
y = features_df['Label']  # Target (activity labels)

# ----- Visualize the features
# Select the features you want to visualize, for example, mean and std of a specific sensor
selected_features = ['back_x_mean', 'back_y_mean', 'back_z_mean', 'thigh_x_mean', 'thigh_y_mean', 'thigh_z_mean',
'back_x_std', 'back_y_std', 'back_z_std', 'thigh_x_std', 'thigh_y_std', 'thigh_z_std',
'back_x_median', 'back_y_median', 'back_z_median', 'thigh_x_median', 'thigh_y_median', 'thigh_z_median',
#'back_x_skewness', 'back_y_skewness', 'back_z_skewness', 'thigh_x_skewness', 'thigh_y_skewness', 'thigh_z_skewness',
#'back_x_kurtosis', 'back_y_kurtosis', 'back_z_kurtosis', 'thigh_x_kurtosis', 'thigh_y_kurtosis', 'thigh_z_kurtosis',
'back_x_iqr', 'back_y_iqr', 'back_z_iqr', 'thigh_x_iqr', 'thigh_y_iqr', 'thigh_z_iqr',
'back_x_rms', 'back_y_rms', 'back_z_rms', 'thigh_x_rms', 'thigh_y_rms', 'thigh_z_rms',
'back_x_energy', 'back_y_energy', 'back_z_energy', 'thigh_x_energy', 'thigh_y_energy', 'thigh_z_energy'
#'back_x_entropy', 'back_y_entropy', 'back_z_entropy', 'thigh_x_entropy', 'thigh_y_entropy', 'thigh_z_entropy'
]
# 
#------------ VISUALISE unbalanced data
# Method a: Boxplot
plt.figure(figsize=(12, 6))
for feature in selected_features:
    sns.boxplot(x='Label', y=feature, data=features_df)
    plt.title(f'Comparison of {feature} across activities')
    plt.show()

# Method b: pairplot
# Only select a few features to avoid overcrowding the plot
pairplot_features = ['back_x_mean', 'thigh_x_mean']
sns.pairplot(features_df[pairplot_features + ['Label']], hue='Label')
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()

# Method c: Histograms    
plt.figure(figsize=(12, 6))
for feature in selected_features:
    for activity in df['label'].unique():
        activity_data = features_df[features_df['Label'] == activity]
        sns.histplot(activity_data[feature], kde=True, label=f'Activity {activity}')
    plt.title(f'Distribution of {feature} across activities')
    plt.legend()
    plt.show()
    
# ------------- CREATE Balanced dataset
# Method 1: Oversampling the Minority Class
from imblearn.over_sampling import RandomOverSampler

# Initialize the oversampler
oversampler = RandomOverSampler(random_state=42)
# Fit and apply the oversampler to the data
X_balanced, y_balanced = oversampler.fit_resample(X, y)

# Check the new distribution
print(Counter(y_balanced))

# ----
# Method 2: Undersampling the Majority Class
from imblearn.under_sampling import RandomUnderSampler

# Initialize the undersampler
undersampler = RandomUnderSampler(random_state=42)

# Fit and apply the undersampler to the data
X_balanced, y_balanced = undersampler.fit_resample(X, y)

# Check the new distribution
print(Counter(y_balanced))

# ----
# Method 3: SMOTE (Synthetic Minority Over-sampling Technique)
from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit and apply SMOTE to the data
X_balanced, y_balanced = smote.fit_resample(X, y)

# Check the new distribution
print(Counter(y_balanced))
# ----------------------------
features_df_balanced = pd.DataFrame(X_balanced)
labels_balanced = np.array(y_balanced)
features_df_balanced['Label'] = labels_balanced

# Method a: Boxplot
plt.figure(figsize=(12, 6))
for feature in selected_features:
    sns.boxplot(x='Label', y=feature, data=features_df_balanced)
    plt.title(f'Comparison of {feature} across activities')
    plt.show()

# Method c: Histograms    
plt.figure(figsize=(12, 6))
for feature in selected_features:
    for activity in df['label'].unique():
        activity_data = features_df_balanced[features_df_balanced['Label'] == activity]
        sns.histplot(activity_data[feature], kde=True, label=f'Activity {activity}')
    plt.title(f'Distribution of {feature} across activities')
    plt.legend()
    plt.show()
    

# ------ Identify relevant features
from sklearn.feature_selection import SelectKBest, f_classif
# -- Method a. Univariate Feature Selection:
# Use statistical tests to select features that have the strongest 
# relationship with the target variable.
#
# Assume X is the feature DataFrame and y is the labels
selector = SelectKBest(score_func=f_classif, k='all')
X=features_df[selected_features]
y=features_df['Label']
selector.fit(X, y)
scores = selector.scores_

# Create a DataFrame for better visualization
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores})
feature_scores = feature_scores.sort_values(by='Score', ascending=False)
print(feature_scores)
 
# -- Method b. Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE

# Initialize the model
model = RandomForestClassifier()

# RFE with a RandomForestClassifier
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X, y)

# Get the ranking of the features
ranking = rfe.ranking_
feature_ranking = pd.DataFrame({'Feature': X.columns, 'Ranking': ranking})
feature_ranking = feature_ranking.sort_values(by='Ranking')
print(feature_ranking)

# RFECV
# TOO LONG
#from sklearn.feature_selection import RFECV
#from sklearn.svm import SVR
#estimator = SVR(kernel="linear")
#selector = RFECV(estimator, step=1, cv=5)
#selector = selector.fit(X, y)

# -- Method c. Tree-Based Feature Importance
# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)
### WAS anderes:

arr=np.array()