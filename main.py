# *********************************************************************************************************************
# Import Statement
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import zipfile
from datetime import datetime
from file_handling import *
from imblearn.over_sampling import SMOTE
from selection import *
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeRegressor
# *********************************************************************************************************************
# Import Data
# Assign data in folder
in_flder = "data_in"


data = open_unknown_csv(os.path.join(in_flder, 'online_shoppers_intention.csv'), ',')
# *********************************************************************************************************************

# Assign output folder
out_fldr = "data_out"

# *********************************************************************************************************************
# Data Wrangling and Cleaning
# Get list of columns
columns = list(data)

# Get the dependent variable and remove from list
y_name = columns[17]
columns.remove(y_name)

# Get list of categorical variables and remove from list
categorical_list_name = list()
for index, i in enumerate(columns):
    if data[i].dtype == "object":
        categorical_list_name.append(i)
    else:
        if index == 0 or index == 2 or index == 4 or index == 11 or index == 12 or index == 13 or index == 14\
                or index == 16:
            categorical_list_name.append(i)
for i in categorical_list_name:
    columns.remove(i)

# Convert target to boolean
# Target is already boolean

# *********************************************************************************************************************

# *********************************************************************************************************************
# One hot encode for each of the categorical datasets
for i in categorical_list_name:
    # Dummy encode variables
    data_dummies = pd.get_dummies(data[i], prefix=str(i))

    # From the created dummies, see which column, by index has the least values and remove to prevent multi-collinearity
    col_mean = 1
    lowest_col = 0
    for index, j in enumerate(data_dummies.mean()):
        if j < col_mean:
            lowest_col = index
            col_mean = j

    # Drop index with smallest dummy amount
    data_dummies.drop(data_dummies.columns[lowest_col], axis=1, inplace=True)

    # Merge dummies with original data
    data = pd.concat([data, data_dummies], axis='columns')

    # Drop original column
    data = data.drop(i, axis=1)
# *********************************************************************************************************************

# *********************************************************************************************************************
# Data Exploration 2
# Create a correlation matrix of all columns
corr = data.corr().unstack().reset_index()

# Drop diagonal from correlation
corr = corr[corr['level_0'] != corr['level_1']]

# Reset index
corr.reset_index(drop=True, inplace=True)

# Sense many columns are one hot encoded, I will drop the correlations that are from the same original column because
# the correlations are always strong between them.
row_deletion = list()
for i in range(len(corr)):
    row_deletion_hold = 0

    # If '_' exists in column name
    underscore_location_0 = corr.loc[i, 'level_0'].find('_')
    underscore_location_1 = corr.loc[i, 'level_1'].find('_')
    if underscore_location_0 != -1:
        if underscore_location_1 != -1:
            if corr.loc[i, 'level_0'][:underscore_location_0] == corr.loc[i, 'level_1'][:underscore_location_1]:
                row_deletion_hold = 1
    row_deletion.append(row_deletion_hold)

# Add row deletion column
corr['row_deletion'] = row_deletion

# Delete Rows from Row Deletion Column and delete column
corr = corr[corr['row_deletion'] != 1]
corr.drop(columns='row_deletion', inplace=True)

# Reset index
corr.reset_index(drop=True, inplace=True)

# Sort by absolute value
corr['corr_abs'] = corr[0].abs()
corr = corr.sort_values(by='corr_abs', ascending=False)

# Drop correlations under 0.5
corr = corr[corr['corr_abs'] >= 0.5]
corr.drop(columns='corr_abs', inplace=True)

# Make correlation matrix
corr_matrix = corr.pivot(index='level_0', columns='level_1', values=0)

# Plot correlation matrix
plt.figure(3)
ax = sns.heatmap(
    corr_matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# *********************************************************************************************************************

# Findings from data exploration.
#
# The highest points of multi-collinearity exists between the operating system and browser type. This is an expected
# correlation. Because of the lack of correlation, I will proceed by using a decision tree based model that can handle
# having 100's of exstenious varibles, otherwise I will have to perform a dimensionality reduction through wither a PCA
# or a model based feature selection.

# *********************************************************************************************************************
# Normalize Values
# Normally, I would Normalize the independent variables at this step as it tends to yield better results, but since I
# will be mostly using decision tree based models, it isn't as necessary since the models are non-parametric.
# *********************************************************************************************************************

# *********************************************************************************************************************
# Remove/Impute Outliers
# Similarly to normalizing values, I would normally impute or remove outliers prior to modeling, however, since I am
# focusing mostly on the decision tree based models, outliers tend to have a negligible on decision tee models due to
# the methods that they perform splits (splits are typically conducted on population proportion instead of values).
# *********************************************************************************************************************

# *********************************************************************************************************************
# Missing Value Handling
# There are no missing values in the dataset. If there was, there are several ways for me to handle missing data
# including data imputation using mean, mode, monte carlo sampling, clustering. If interested, ask me about missing
# Data handling as it was a focus of mine during my second masters program.
#
# Note: In this case, I would consider just accepting the missing data.
# *********************************************************************************************************************


# *********************************************************************************************************************
# Partition Data
# Separate the data in Train, Test, and Validation Partitions (50, 30, 20)
# Separate data into X and y and split the data into Train and Test/Validation Partitions
X = data.drop(columns=[y_name], axis=1)
y = data[y_name]
X_train, X_test_validation, y_train, y_test_validation = train_test_split(X, y, test_size=.50, random_state=0)

# Separate split the data into Test and Validation Partitions
X_test, X_validation, y_test, y_validation = train_test_split(X_test_validation, y_test_validation, test_size=.40,
                                                              random_state=0)
# *********************************************************************************************************************

# Looking at the distribution of the target
print()
print("Minority fraction of Target: " + str(round(data[y_name].mean(), 4)))

# *********************************************************************************************************************
# Model 1 - Initial. Decision Tree, without over-sampling. Aggressive Pruning to avoid over-fitting
print()
print("Model 1 - Initial Model")
print("Model Type: Regression Based Decision Tree")
# Start Timer
time_strt = datetime.now().replace(microsecond=0)

# Define Decision Tree Algorithm
regressor_1 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=120)

# Run Decision Tree
regressor_1.fit(X_train, y_train)

# Score the partitions
y_train_1_pred = regressor_1.predict(X_train)
y_test_1_pred = regressor_1.predict(X_test)

# Metrics
fpr_1_train, tpr_1_train, thresholds_1_train = metrics.roc_curve(y_train, y_train_1_pred)
fpr_1_test, tpr_1_test, thresholds_1_test = metrics.roc_curve(y_test, y_test_1_pred)

# AUC
print("AUC: " + str(metrics.auc(fpr_1_test, tpr_1_test)))

# Export visualization of tree
tree.export_graphviz(regressor_1, out_file=os.path.join(out_fldr, "Model_1_tree.dot"),
                     feature_names=list(X_train.columns.values))

# Timer End - Print Time
print("Time to complete model: " + str(datetime.now().replace(microsecond=0) - time_strt))
# *********************************************************************************************************************

# *********************************************************************************************************************
# Model 2 - Decision Tree with over-sampling. Aggressive Pruning to avoid over-fitting.
# Considering that there is a sparse amount of target values, I will over-sample where the target = 1.
print()
print("Model 2 - Over-sampling")
print("Model Type: Regression Based Decision Tree")
# Start Timer
time_strt = datetime.now().replace(microsecond=0)

# Over-sample Target = 1
sm = SMOTE(random_state=12, sampling_strategy='auto')
X_train_resampled, y_train_resampled = sm.fit_sample(X_train, y_train)
print("Oversampled minority class fraction of Target: " + str(round(y_train_resampled.mean(), 4)))

# Define Decision Tree Algorithm
regressor_2 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=120)

# Run Decision Tree
regressor_2.fit(X_train_resampled, y_train_resampled)

# Score the partitions
y_train_2_pred = regressor_2.predict(X_train_resampled)
y_test_2_pred = regressor_2.predict(X_test)

# Metrics
fpr_2_train, tpr_2_train, thresholds_2_train = metrics.roc_curve(y_train_resampled, y_train_2_pred)
fpr_2_test, tpr_2_test, thresholds_2_test = metrics.roc_curve(y_test, y_test_2_pred)

# AUC
print("AUC: " + str(metrics.auc(fpr_2_test, tpr_2_test)))

# Export visualization of tree
tree.export_graphviz(regressor_2, out_file=os.path.join(out_fldr, "Model_2_tree.dot"),
                     feature_names=list(X_train.columns.values))

# Timer End - Print Time
print("Time to complete model: " + str(datetime.now().replace(microsecond=0) - time_strt))
# *********************************************************************************************************************.

# Analysis of Model 2. While there is a significant imbalance in the dependent variable and oversampling is a staple to
# improve a model's performance in those situations, using SMOTE had a slight decrease in the overall model's
# performance.

# *********************************************************************************************************************
# Model 3 - Less Pruning. Decision Tree, without over-sampling.
print()
print("Model 3 - Less Pruning")
print("Model Type: Regression Based Decision Tree")
# Start Timer
time_strt = datetime.now().replace(microsecond=0)

# Define Decision Tree Algorithm
regressor_3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=250)

# Run Decision Tree
regressor_3.fit(X_train, y_train)

# Score the partitions
y_train_3_pred = regressor_3.predict(X_train)
y_test_3_pred = regressor_3.predict(X_test)

# Metrics
fpr_3_train, tpr_3_train, thresholds_3_train = metrics.roc_curve(y_train, y_train_3_pred)
fpr_3_test, tpr_3_test, thresholds_3_test = metrics.roc_curve(y_test, y_test_3_pred)

# AUC
print("AUC: " + str(metrics.auc(fpr_3_test, tpr_3_test)))

# Export visualization of tree
tree.export_graphviz(regressor_3, out_file=os.path.join(out_fldr, "Model_3_tree.dot"),
                     feature_names=list(X_train.columns.values))

# Timer End - Print Time
print("Time to complete model: " + str(datetime.now().replace(microsecond=0) - time_strt))
# *********************************************************************************************************************

# Analysis of Model 3.

# *********************************************************************************************************************
# Model 4 - Boosted Decision Tree.
print()
print("Model 4 - Boosted Decision Tree")
print("Model Type: Boosted Regression Based Decision Tree")
# Start Timer
time_strt = datetime.now().replace(microsecond=0)

# Define Decision Tree Algorithm
boosted_regressor = GradientBoostingRegressor(max_depth=3)

# Run Decision Tree
boosted_regressor.fit(X_train, y_train)

# Score the partitions
y_train_4_pred = boosted_regressor.predict(X_train)
y_test_4_pred = boosted_regressor.predict(X_test)

# Metrics
fpr_4_train, tpr_4_train, thresholds_4_train = metrics.roc_curve(y_train, y_train_4_pred)
fpr_4_test, tpr_4_test, thresholds_4_test = metrics.roc_curve(y_test, y_test_4_pred)

# AUC
print("AUC: " + str(metrics.auc(fpr_4_test, tpr_4_test)))

# Export visualization of tree
# Unfortunately, you can't show a graph of an ensemble decision tree... very easily...

# Timer End - Print Time
print("Time to complete model: " + str(datetime.now().replace(microsecond=0) - time_strt))
# *********************************************************************************************************************

# Analysis of Model 4.

# *********************************************************************************************************************
# Plot ROC Curves with matplotlib - Train
plt.figure(8)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_1_train, tpr_1_train, label='Model 1 - Initial')
plt.plot(fpr_2_train, tpr_2_train, label='Model 2 - Over-Sampling')
plt.plot(fpr_3_train, tpr_3_train, label='Model 3 - Less Pruning')
plt.plot(fpr_4_train, tpr_4_train, label='Model 4 - Boosted Decision Tree')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - Train')
plt.legend(loc='best')
plt.show()

# Plot ROC Curves with matplotlib - Test
plt.figure(9)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_1_test, tpr_1_test, label='Model 1 - Initial')
plt.plot(fpr_2_test, tpr_2_test, label='Model 2 - Over-Sampling')
plt.plot(fpr_3_test, tpr_3_test, label='Model 3 - Less Pruning')
plt.plot(fpr_4_test, tpr_4_test, label='Model 4 - Boosted Decision Tree')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - Test')
plt.legend(loc='best')
plt.show()
# *********************************************************************************************************************

# Analysis of the 5 model's ROC curve.

# *********************************************************************************************************************
# Score Validation Partition and analyse ROC Curves.

# Score the validation partitions
y_valid_1_pred = regressor_1.predict(X_validation)
y_valid_2_pred = regressor_2.predict(X_validation)
y_valid_3_pred = regressor_3.predict(X_validation)
y_valid_4_pred = boosted_regressor.predict(X_validation)

# Metrics for validation partitions
fpr_1_valid, tpr_1_valid, thresholds_1_valid = metrics.roc_curve(y_validation, y_valid_1_pred)
fpr_2_valid, tpr_2_valid, thresholds_2_valid = metrics.roc_curve(y_validation, y_valid_2_pred)
fpr_3_valid, tpr_3_valid, thresholds_3_valid = metrics.roc_curve(y_validation, y_valid_3_pred)
fpr_4_valid, tpr_4_valid, thresholds_4_valid = metrics.roc_curve(y_validation, y_valid_4_pred)

# Plot ROC Curves with matplotlib
plt.figure(10)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_1_valid, tpr_1_valid, label='Model 1 - Initial')
plt.plot(fpr_2_valid, tpr_2_valid, label='Model 2 - Over-Sampling')
plt.plot(fpr_3_valid, tpr_3_valid, label='Model 3 - Less Pruning')
plt.plot(fpr_4_valid, tpr_4_valid, label='Boosted Decision Tree')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - Validation')
plt.legend(loc='best')
plt.show()
# *********************************************************************************************************************

# Analysis of Validation Partition.

# *********************************************************************************************************************
# Export validation Preditions
# Create a dataframe that has the validation partition pred vs actual
valid_actual_vs_predition = pd.DataFrame({'Actual': y_valid_4_pred, 'Predicted': y_valid_4_pred})

# Export Predictions
valid_actual_vs_predition.to_csv(os.path.join(out_fldr, 'predictions.csv'))
# *********************************************************************************************************************
