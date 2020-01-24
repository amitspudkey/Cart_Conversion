# *********************************************************************************************************************
# Import Statement
from joblib import load
import pandas as pd
import os
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeRegressor
from file_handling import *
# *********************************************************************************************************************
# Assign output folder
out_fldr = "data_out"
# *********************************************************************************************************************
# Import Data
data_location = select_file_in()
data = open_unknown_csv(data_location, ',')

out_location = select_file_out_csv("Select Output location")
# *********************************************************************************************************************
# Make Predictions
model = load(os.path.join(out_fldr, "Model_3.joblib"))
predictions = model.predict(data)

data["Prediction"] = predictions
# *********************************************************************************************************************
# Write Data
data.to_csv(out_location)
# *********************************************************************************************************************
