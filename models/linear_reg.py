import pandas as pd
# from sqlalchemy import create_engine

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt


#  Load data from CSV files to the in-memory database


# class Data:
#     def __init__(self):
#         
#         self.engine = create_engine('sqlite://', echo=False)
        
#         filenames = [
#             'genome-tags',
#             'genome-scores',
#             'movies',
#             'tags_shuffled_rehashed',
#             'train_ratings',
#             'val_ratings'
#         ]

#         for filename in filenames:
#             pd.read_csv('../data/' + filename + '.csv', nrows=3000).to_sql(filename.replace('-', '_'), con=self.engine)

# data = Data()
# print(data.engine.execute('SELECT * from movies').fetchone())



# Process Data

train_ratings = pd.read_csv('../data/train_ratings.csv')
movies = pd.read_csv('../data/movies.csv')
val_rating = pd.read_csv('../data/val_ratings.csv')
train_ratings = train_ratings.append(val_rating)

# It only has genres as training features.
all_train_df = pd.merge(train_ratings, movies, on='movieId')
all_train_df = pd.concat([all_train_df.drop('genres', axis=1), all_train_df['genres'].str.get_dummies(sep='|')], axis=1)

if '(no genres listed)' in all_train_df.columns:
    all_train_df = all_train_df.drop([ 'userId', 'movieId', 'title', '(no genres listed)'], axis=1)
else:
    all_train_df = all_train_df.drop([ 'userId', 'movieId', 'title'], axis=1)



# Train Model


# train_df = train_df[train_df['userId'] == 1]

# print("start training...")

# users_lm = {}
# all_train_df.set_index(keys=['userId'], drop=False,inplace=True)
# users_all = all_train_df['userId'].unique().tolist()
# for u in users_all:
#     user = all_train_df.loc[all_train_df.userId == u]
#     train_X = user.drop(['userId','rating'], axis=1)
#     train_y = user['rating']
#     reg = LinearRegression().fit(train_X, train_y)
#     users_lm[u] = reg
# print("train all users finish")


# Get the training dataset
train_y = all_train_df['rating']
train_X = all_train_df.drop('rating', axis=1)

print("start training...")

# Train the linear model
reg = LinearRegression().fit(train_X, train_y)
print('Fitted coefficients: ', reg.coef_)

######################################################
# Validation/Test Model

# Get the validation dataset
# val_ratings = pd.read_csv('../data/test_ratings.csv')
# val_df = pd.merge(val_ratings, movies, on='movieId')
# val_df = pd.concat([val_df.drop('genres', axis=1), val_df['genres'].str.get_dummies(sep='|')], axis=1)
# #val_df = val_df[val_df['userId'] == 1]
# if '(no genres listed)' in val_df.columns:
#     val_df = val_df.drop(['userId', 'movieId', 'title', '(no genres listed)'], axis=1)
# else:
#     val_df = val_df.drop(['userId', 'movieId', 'title'], axis=1)
# val_y = val_df['rating']
# val_X = val_df.drop('rating', axis=1)
# print(val_X)
# print(val_y)

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
pred_y = reg.predict(test_df)
df = pd.DataFrame(pred_y, columns=["rating"])
df.to_csv('../data/lreg.csv', index_label = 'Id')
# rms = sqrt(mean_squared_error(val_y, pred_y))
# print('RMSE: ', rms)

######################################################
# Results
######################################################

# Fitted coefficients:  [-3.45982143e-02 -5.50223214e-01  2.77555756e-17 -3.33066907e-16
#  -2.44866071e-01 -1.34263393e+00 -2.22044605e-16  8.65625000e-01
#   2.22044605e-16  0.00000000e+00  0.00000000e+00  0.00000000e+00
#   0.00000000e+00 -4.52008929e-01 -8.98437500e-01  2.23883929e-01
#  -2.35937500e-01 -8.98437500e-01 -6.08705357e-01]
# RMSE:  1.022785329609021


# all users training:
# Fitted coefficients:  [-0.16610356  0.09187795  0.28755384 -0.36053963 -0.12494848  0.2356944
#   0.28587089  0.16818677  0.07146438  0.26920447 -0.19248269  0.0782627
#   0.09264103  0.13665857  0.00772457  0.03662268 -0.06340973  0.30812347
#   0.03430783]
# RMSE:  0.8792970739852456
