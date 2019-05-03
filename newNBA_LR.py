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
import math

# CSV path
# CSV_PATH = "/Users/khoatran/Desktop/Repositories/NBA_MVP_PREDICTOR/newNBA_Data.csv"
CSV_PATH = "/Users/khoatran/Desktop/reducedNBA.csv"
# CSV_PATH = "/Users/khoatran/Desktop/ml-mvp-predict/end-of-season/final-csv-data/historical-mvps.csv"

def main():
    # Load Data
    dataset = pd.read_csv(CSV_PATH)
    print "Observation Count: ", len(dataset)

    # Print top 10 data using pandas lib
    # to test if data has been loaded successfully
    dataset = dataset.fillna(dataset.mean())
    print dataset.head(10)

    # Extract column names
    column_names = list(dataset.columns.values)
    print "Dataset Column Names: {column_names}".format(column_names=column_names)

    # Selecting features
    # Not including ...'Name', 'Season', 'Team', 'Team Wins','Position'

    # features = ['Team Win Percentage', 'MIN', 'PTS','FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB','AST',\
    #             'STL', 'BLK', 'TOV', 'PER', 'OFFRTG', 'DEFRTG','NETRTG', 'AST_PCT', 'AST_TO_TO',\
    #             'AST_RATIO', 'TO_RATIO', 'EFG', 'TS', 'USG_RATE', 'WIN_SHARE',\
    #             'Team_Valuation', 'Team_Home_Attendance_Percentage', 'Salary(millions)']
    # features = ['Team Win Percentage', 'PTS','REB','AST','PER','WIN_SHARE']
    
    # features = ['Rank', 'Age', 'Pts Won', 'Pts Max',\
    #  'Team Wins', 'Overall Seed', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%',\
    #   'FT%', 'WS', 'WS/48', 'VORP', 'BPM']

    # features = ['Team Wins', 'Overall Seed', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%',\
    #   'FT%', 'WS', 'WS/48', 'VORP', 'BPM']
    features = ['Team Win Percentage', 'PTS',\
            'PER', 'NETRTG', 'AST_TO_TO',\
            'USG_RATE', 'WIN_SHARE','AST','REB','FG_PCT']

    # Selecting target (Y)
    # target = 'PTS_WON2'
    target = 'PTS_WON2'
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
    df = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred}) 
    print df
    # The coefficients
    LR_coef = LR_model.coef_
    print 'Coefficients:'
    print LR_coef
    i = 0
    while(i<len(features)):
        print str(features[i])+ "\t" + str(LR_coef[i])
        i+=1

    mse = mean_squared_error(test_y, y_pred)
    rmse = math.sqrt(mse)

    # The mean squared error
    print("Mean squared error: %.2f"
          % mse)
    # The root mean squared error
    print("Root mean squared error: %.2f"
          % rmse)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test_y, y_pred))

    print "Giannis"
    G = [[0.732,27.7,30.9,12.8,1.58,31,14.4,5.9,12.5,.578]]
    print LR_model.predict(G)
    print "Harden"
    H = [[.646,36.1,30.6,6.4,1.51,39.3,15.2,7.5,6.6,.442]]
    print LR_model.predict(H)
    print "Jokic"
    J = [[.659,20.1,26.3,6.3,2.34,26.8,11.8,7.3,10.8,.511]]
    print LR_model.predict(J)
    print "Shaq"
    Shaq = [[.634,22.9,27.03,10.5,.99,30,11,2.7,10.4,.601]]
    print LR_model.predict(Shaq)
    print "Steve Nash"
    S = [[.658,15.5,22.04,12.8,3.51,20,10.9,11.5,3.3,.502]]
    print LR_model.predict(S)

    # from sklearn.preprocessing import PolynomialFeatures
    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.linear_model import Ridge
    # from sklearn.linear_model import Lasso

    # steps = [
    #     ('scalar', StandardScaler()),
    #     ('poly', PolynomialFeatures(degree=2)),
    #     ('model', Lasso(alpha=0.2, fit_intercept=True))
    # ]

    # pipeline = Pipeline(steps)

    # pipeline.fit(train_x, train_y)
    # print('Training score2: {}'.format(pipeline.score(train_x, train_y)))
    # print('Test score2: {}'.format(pipeline.score(test_x, test_y)))

if __name__ == "__main__":
    main()
