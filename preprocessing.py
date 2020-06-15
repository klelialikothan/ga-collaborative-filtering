import pandas as pd
import numpy as np

# Basic variables
N: int = 943
M: int = 1682
col_names = ['user id', 'movie id', 'rating', 'timestamp']

# Import file to dataframe
print('Importing and processing data.')
df = pd.read_table('u.data', names=col_names, usecols=col_names[0:3], dtype=np.int32)

# Process ratings and save to file
user_ratings = np.zeros([N, M], dtype=np.int32)

for i in range(0, N):   # foreach user in dataset
    u = df.loc[df['user id'] == (i+1)]
    temp = np.array([k for j, k in zip(u['movie id'], u['rating'])])
    x = 0

    for j, k in zip(u['movie id'], u['rating']):
        # store (existing) ratings in array row, filling empty cells with 0
        user_ratings[i, (j - 1)] = temp[x]
        x += 1
df.drop(columns=col_names[0:3])
np.save('user_ratings.npy', user_ratings)

# Read test data from file, remove timestamps and save results as ndarray to file
df = pd.read_table('ua.test', names=col_names, usecols=col_names[0:3], dtype=np.int32)
np.save('test_data.npy', df.to_numpy())

# Construct train data by removing test set data from user_ratings
for i in range(0, N):   # foreach user in dataset
    u = df.loc[df['user id'] == (i+1)]
    for j, k in zip(u['movie id'], u['rating']):
        # replace designated test ratings with 0
        user_ratings[i, (j - 1)] = 0
np.save('train_data.npy', user_ratings)

print('Done.')
