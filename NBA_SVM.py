import csv
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
# CSV path
CSV_PATH = "/Users/khoatran/Desktop/finalNBA_Data.csv"

def main():
	# Load Data
    dataset = pd.read_csv(CSV_PATH)
    print "Observation Count: ", len(dataset)

    # Print top 10 data using pandas lib
    # to test if data has been loaded successfully
    dataset = dataset.fillna(dataset.mean())
    # print dataset.head(10)

    # Extract column names
    # column_names = list(dataset.columns.values)
    # print "Dataset Column Names: {column_names}".format(column_names=column_names)

    # Selecting features
    # Not including ...'Name', 'Season', 'Team', 'Team Wins','Position'

    features24 = ['Team Win Percentage', 'MIN', 'PTS','FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB','AST',\
                'STL', 'BLK', 'TOV', 'PER', 'OFFRTG', 'DEFRTG','NETRTG', 'AST_PCT', 'AST_TO_TO',\
                'AST_RATIO', 'TO_RATIO', 'EFG', 'TS', 'USG_RATE', 'WIN_SHARE',\
                'Team_Valuation', 'Team_Home_Attendance_Percentage', 'Salary(millions)']

    FIfeatures = ['Team Win Percentage', 'PTS',\
            'PER', 'NETRTG', 'AST_TO_TO',\
            'USG_RATE', 'WIN_SHARE','AST','REB','FG_PCT']
    features10 = ['Team Win Percentage','PER', 'NETRTG', 'AST_TO_TO',\
            'USG_RATE', 'WIN_SHARE','TS', 'VORP', 'BPM', 'Salary(millions)']

    # Selecting target (Y)
    target = 'Share'
    # splitting training and testing data (70:30)
    train_x, test_x, train_y, test_y = train_test_split(dataset[features10], dataset[target], train_size=0.7)

	# Train SVC
    # Kernel = Linear
    SVC_model_linear = SVC(kernel='linear')
    SVC_model_linear.fit(train_x,train_y)
	# Calculate Accuracy
    SVC_accuracy_train_linear = SVC_model_linear.score(train_x,train_y)
    SVC_accuracy_test_linear = SVC_model_linear.score(test_x, test_y)
    print "Train Accuracy (Linear): ", SVC_accuracy_train_linear
    print "Test Accuracy (Linear): ", SVC_accuracy_test_linear

if __name__ == "__main__":
    main()