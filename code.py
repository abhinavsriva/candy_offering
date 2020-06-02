import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing, linear_model, metrics


## Loading csv file into a dictionary/dataframe
################################################
df = pd.read_csv("../candy-data.csv")


## Sorting and printing data
###########################################


# ldf = df.sort_values(by=['winpercent'], ascending=False)
# (ldf.head(20)).to_latex('Sorted.tex',
#                         columns=['competitorname','winpercent'],index=False)


## Computing pointbiserial coorelation coefficient that helps
## to coorelate between binary and continuous variables. Also
## plotting histogram to visually inspect data


cols = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy',
        'nougat', 'crispedricewafer', 'hard', 'bar',
        'pluribus']

r = np.zeros(shape=(len(cols), 1))
p = np.zeros(shape=(len(cols), 1))

dummy_df = pd.DataFrame()


print('Point biserial coefficient')
print('---------------------------')

for i, s in enumerate(cols):
    no = df[(df[s] == 0)]
    yes = df[(df[s] == 1)]
    r[i], p[i] = stats.pointbiserialr(df[s].values, df['winpercent'].values)

    print(s, r[i], p[i])

    ## Saving results in Data-frame
    # dummy_df.loc[i,0] = s;
    # dummy_df.loc[i,1] = r[i]
    # dummy_df.loc[i,2] = p[i]
    
    #plt.hist(no.winpercent, 10, facecolor='b', alpha=0.5)
    #plt.hist(yes.winpercent, 10, facecolor='r',alpha=0.5)
    # plt.xlabel('winpercent)
    # plt.ylabel('i')
    # plt.show()

### Printing resutls to a latex file
# dummy_df.to_latex('pbico.text',index=False)

print("=================================\n")




## Computing Pearson coefficient between two continuous variables
## and drawing scatter plots to visually inspect data

cols = ['chocolate', 'sugarpercent', 'pricepercent']

print('Pearson coefficient')
print('-------------------------')


for i in cols:
    r, p = stats.pearsonr(df[i].values, df['winpercent'].values)
    print(i, '\t', r, p)

#     plt.scatter(df[i].values, df['winpercent'].values, c='b')
#     plt.xlabel('winpercent')
#     plt.ylabel('i')
#     plt.show()

print('=========================================\n')



## List of necessary columns

cols = list(df.columns)
cols.remove('competitorname')
cols.remove('winpercent')

## Normalizing values

x = df[cols].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled)
df[cols] = df_temp

# ## Linear regression on cols = ['chocolate', 'fruity', 'caramel',
# ## 'peanutyalmondy', 'nougat', 'crispedricewafer', 'hard',
# ## 'bar', 'pluribus', 'pricepercent']


## Model generation using Linear Regression

r = linear_model.LinearRegression()
x_train = np.asanyarray(df[cols])
y_train = np.asanyarray(df[['winpercent']])
r.fit(x_train, y_train)

print('Regression coefficients =', r.coef_, '\n Intercept = ', r.intercept_, '\n')

# ## Predicting and computing r-squared

x_test = np.asanyarray(df[cols])
y_hat = r.predict(x_test)
y_test = np.asanyarray(df[['winpercent']])

print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_hat)))
print("R2-score", metrics.r2_score(y_test, y_hat),'\n')




## For more detailed analysis

# equation = 'winpercent ~'
# for i in cols:
#     equation = equation + i
#     if i != cols[len(cols) - 1]:
#         equation += ' + '
#
# import statsmodels.formula.api as sm
#
# model = sm.ols(formula=equation, data=df)
# fitted = model.fit()
# print(fitted.summary())

## Printing results for latex
# l = fitted.summary().tables
# print_sum = pd.DataFrame(l[1])
# print_sum.to_latex('Summary.tex')

cols.remove('caramel')
cols.remove('bar')
cols.remove('pluribus')
cols.remove('nougat')
cols.remove('pricepercent')


equation = 'winpercent ~'
for i in cols:
    equation = equation + i
    if i != cols[len(cols) - 1]:
        equation += ' + '

import statsmodels.formula.api as sm

model = sm.ols(formula=equation, data=df)
fitted = model.fit()
print(fitted.summary())

## Printing results for latex
# l = fitted.summary().tables
# print_sum = pd.DataFrame(l[1])
# print_sum.to_latex('Summary2.tex')

