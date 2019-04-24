import csv
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# CSV path
CSV_PATH = "/Users/khoatran/Desktop/Repositories/NBA_MVP_PREDICTOR/NBA_Data.csv"


# Load Data
dataset = pd.read_csv(CSV_PATH)
print "Observation Count: ", len(dataset)

# Print top 10 data using pandas lib
# to test if data has been loaded successfully
print dataset.head(10)

# Extract column names
column_names = list(dataset.columns.values)
print "Dataset Column Names: {column_names}".format(column_names=column_names)

# Selecting features
# Not including ...Team, Team wins, position, etc

features = ['Team Win Percentage','MIN','PTS','FG_PCT','FG3_PCT','FT_PCT','REB','AST',\
    'STL','BLK','TOV','PER','OFFRTG','DEFRTG','NETRTG','AST_PCT',\
    'AST_TO_TO','AST_RATIO','TO_RATIO','EFG','TS',\
    'USG_RATE','WIN_SHARE']
target = 'PTS_WON'
# splitting training and testing data (70:30)
train_x, test_x, train_y, test_y = train_test_split(dataset[features], dataset[target], train_size=0.7)

print "train_x size: ", len(train_x)
print "train_y size: ", len(train_y)

print "test_x size: ", len(test_x)
print "test_y size: ", len(test_y)

# Fit model
LR_model = LinearRegression()
LR_model.fit(train_x,train_y)

# Calculate Accuracy
LR_accuracy_train = LR_model.score(train_x, train_y)
LR_accuracy_test  = LR_model.score(test_x, test_y)
print "Train Accuracy: ", LR_accuracy_train
print "Test Accuracy : ", LR_accuracy_test

# Make predictions using the testing set
y_pred = LR_model.predict(test_x)

# The coefficients
print('Coefficients: \n', LR_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_y, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_y, y_pred))

# # Plot outputs
# plt.scatter(test_x, test_y,  color='black')
# plt.plot(test_x, y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()

