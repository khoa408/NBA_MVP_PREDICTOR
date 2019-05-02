import pandas as pd
import numpy as np
import plotly.plotly as py
from sklearn.preprocessing import StandardScaler

plotly.tools.set_credentials_file(username='khoatran1997', api_key='0FE1TzylMIG4iSjc3KP7')


# import plotly 
# plotly.tools.set_credentials_file(username='khoatran1997', api_key='0FE1TzylMIG4iSjc3KP7')

# CSV path
CSV_PATH = "/Users/khoatran/Desktop/Repositories/NBA_MVP_PREDICTOR/NBA_Data.csv"


# Load Data
df = pd.read_csv(CSV_PATH)
print "Observation Count: ", len(df)

df.columns=['Class','Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash','Magnesium',\
            'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols','Proanthocyanins',\
            'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline']

df.columns=['PTS_WON','Team Win Percentage','MIN','PTS','FG_PCT','FG3_PCT','FT_PCT','REB','AST',\
        'STL','BLK','TOV','PER','OFFRTG','DEFRTG','NETRTG','AST_PCT',\
        'AST_TO_TO','AST_RATIO','TO_RATIO','EFG','TS',\
        'USG_RATE','WIN_SHARE']

df.tail()

# split data table into data X and class labels y

X = df.iloc[:,1:].values
y = df.iloc[:,0].values

X_std = StandardScaler().fit_transform(X)

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# List of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples descending order
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,25)],
    y=var_exp,
    name='Individual'
)

trace2 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,25)], 
    y=cum_var_exp,
    name='Cumulative'
)

data = [trace1, trace2]

layout=dict(
    title='Variance by different PC\'s',
    yaxis=dict(
        title='Variance in percent(%)'
    ),
    annotations=list([
        dict(
            x=1.16,
            y=1.05,
            xref='paper',
            yref='paper',
            text='Explained Variance',
            showarrow=False,
        )
    ])
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='selecting-principal-components')

matrix_w = np.hstack((eig_pairs[0][1].reshape(25,1), 
                      eig_pairs[1][1].reshape(25,1)))

print('Matrix W:\n', matrix_w)
