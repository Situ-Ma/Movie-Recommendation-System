from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

import pandas as pd
# from sqlalchemy import create_engine

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt


#  Load and pre-process data

train_ratings = pd.read_csv('../data/train_ratings.csv')
movies = pd.read_csv('../data/movies.csv')
val_rating = pd.read_csv('../data/val_ratings.csv')
train_ratings = train_ratings.append(val_rating)

# It only has genres as training features. Just for now.
all_train_df = pd.merge(train_ratings, movies, on='movieId')
all_train_df = pd.concat([all_train_df.drop('genres', axis=1), all_train_df['genres'].str.get_dummies(sep='|')], axis=1)

if '(no genres listed)' in all_train_df.columns:
    all_train_df = all_train_df.drop([ 'userId', 'movieId', 'title', '(no genres listed)'], axis=1)
else:
    all_train_df = all_train_df.drop([ 'userId', 'movieId', 'title'], axis=1)
#train_df = train_df[train_df['userId'] == 1]



# Train Model


# Get the training dataset
train_y = all_train_df['rating']
train_X = all_train_df.drop('rating', axis=1)

print("start training...")

# Train the Random Forest model
clf = RandomForestRegressor(n_estimators=300).fit(train_X, train_y)


# Test Model

test_df = pd.read_csv('../data/test_ratings.csv')
test_df = pd.merge(test_df, movies, on='movieId', how='left')
test_df = pd.concat([test_df.drop('genres', axis=1), test_df['genres'].str.get_dummies(sep='|')], axis=1)
#val_df = val_df[val_df['userId'] == 1]
if '(no genres listed)' in test_df.columns:
    test_df = test_df.drop(['Id','userId', 'movieId', 'title', '(no genres listed)'], axis=1)
else:
    test_df = test_df.drop(['Id','userId', 'movieId', 'title'], axis=1)


# Output

# save results to compare models
pred_y = clf.predict(test_df)
df = pd.DataFrame(pred_y, columns=["rating"])
df.to_csv('../data/rf.csv', index_label = 'Id')
# rms = sqrt(mean_squared_error(val_y, pred_y))
# print('RMSE: ', rms)
