import numpy as np
import pandas as pd

import time


import heapq
from operator import itemgetter

from numba import jit

# This method is modified based on the factor method at https://hub.packtpub.com/recommending-movies-scale-python/
@jit(nopython=True)
def matrix_factorization(ratings, k, steps, alpha, beta):
    N, M = ratings.shape
    U = np.random.rand(N, k)
    V = np.random.rand(k, M)

    for _ in range(steps):
        for i in range(N):
            for j in range(M):
                if ratings[i,j] > 0:
                    eij = ratings[i,j] - np.dot(U[i,:], V[:,j])
                    for k in range(k):
                        U[i,k] += alpha * (2 * eij * V[k,j]- beta * U[i,k])
                        V[k,j] += alpha * (2 * eij * U[i,k]- beta * V[k,j])

        e = 0

        for i in range(N):
            for j in range(M):
                if ratings[i,j] > 0:
                    e += pow(ratings[i,j] - np.dot(U[i,:], V[:,j]), 2)
                    for k in range(k):
                        e += (beta/2) * (pow(U[i,k], 2) + pow(V[k,j], 2))
        
        if e < 0.001:
            break
        
    return np.dot(U, V)

# # Recommendation: ranking of movies for user
# def predict_ranking(user, movie):
#            uidx = users.index(user)
#            midx = movies.index(movie)
#            return model[uidx, midx]

# # recommendation: return six highest rating movies
# def top_rated(user, n=6):
#     movies = [(mid, predict_ranking(user, mid)) for mid in movies]
#         return heapq.nlargest(n, movies, key=itemgetter(1))

def main():
    # Load train dataset
    train_ratings = pd.read_csv('../data/train_ratings.csv')
    # train_ratings = train_ratings[:100]
    # mean = train_ratings['rating'].mean()

    # # Convert the train dataset to a N-by-M matrix
    # # N: Number of users
    # # M: Number of movies
    start_time = time.time()
    ratings = train_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).to_numpy()
    end_time = time.time()
    print('Time for building a N-by-M matrix: ', end_time - start_time)

    users, movies = set(train_ratings['userId']), set(train_ratings['movieId'])
    user_indexes, movie_indexes = dict(), dict()

    for i, user in enumerate(sorted(users)):
        user_indexes[user] = i

    for i, movie in enumerate(sorted(movies)):
        movie_indexes[movie] = i

    # Nested function for predicting ratings
    def predict_rating(row):
        if row['userId'] not in users or row['movieId'] not in movies:
            return 0.0
        return model[user_indexes[row['userId']], movie_indexes[row['movieId']]]

    # cross validation
    betas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

    for beta in betas:
        # Training the model by matrix factorization
        start_time = time.time()
        model = matrix_factorization(ratings, 8, 5000, 0.0002, beta)
        end_time = time.time()
        print('Time for matrix factorization: ', end_time - start_time)

        # Load test dataset
        test_ratings = pd.read_csv('../data/test_ratings.csv')
        test_ratings = test_ratings[:1]

        # Predicting ratings
        start_time = time.time()
        test_ratings['rating'] = test_ratings.apply(lambda row: predict_rating(row), axis=1)
        end_time = time.time()
        print('Time for predicting ratings: ', end_time - start_time)

        # Ratings post-processing
        test_ratings['rating'] = test_ratings['rating'].apply(lambda x: x if x != 0.0 else mean)
        test_ratings['rating'] = test_ratings['rating'].apply(lambda x: x if x <= 5.0 else 5.0)

        # print results
        test_ratings = test_ratings.drop(['Id', 'userId', 'movieId'], axis=1)
        file_name = '../data/result_' + str(beta) + '.csv'
        test_ratings.to_csv(file_name, index_label='Id')

    

if __name__ == '__main__':
    main()