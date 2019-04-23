import random
import pandas as pd
import numpy as np

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

#-------------------------
# LOAD AND PREP THE DATA
#-------------------------
 
raw_data = pd.read_csv(r'..\data\lastfm-360k\usersha1-artmbid-artname-plays.tsv', header=None,
    names=['user-mboxsha1', 'musicbrainz-artist-id', 'artist-name', 'plays'], sep='\t')
raw_data = raw_data.drop(raw_data.columns[1], axis=1)
raw_data.columns = ['user', 'artist', 'plays']
 
# Drop rows with missing values
data = raw_data.dropna()

# Convert artists names into numerical IDs
data['user_id'] = data['user'].astype("category").cat.codes
data['artist_id'] = data['artist'].astype("category").cat.codes

# Create a lookup frame so we can get the artist names back in 
# readable form later.
item_lookup = data[['artist_id', 'artist']].drop_duplicates()
item_lookup['artist_id'] = item_lookup.artist_id.astype(str)

data = data.drop(['user', 'artist'], axis=1)

# Drop any rows that have 0 plays
data = data.loc[data.plays != 0]

# Create lists of all users, artists and plays
users = list(np.sort(data.user_id.unique()))
artists = list(np.sort(data.artist_id.unique()))
plays = list(data.plays)

# Get the rows and columns for our new matrix
rows = data.user_id.astype(int)
cols = data.artist_id.astype(int)
 
# Contruct a sparse matrix for our users and items containing number of plays
data_sparse = sparse.csr_matrix((plays, (rows, cols)), shape=(len(users), len(artists)))


def nonzeros(m, row):
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]
      
      
def implicit_als_cg(Cui, features=20, iterations=20, lambda_val=0.1):
    user_size, item_size = Cui.shape

    X = np.random.rand(user_size, features) * 0.01
    Y = np.random.rand(item_size, features) * 0.01

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

    for iteration in range(iterations):
        print('iteration %d of %d' % (iteration+1, iterations))
        least_squares_cg(Cui, X, Y, lambda_val)
        least_squares_cg(Ciu, Y, X, lambda_val)
    
    return sparse.csr_matrix(X), sparse.csr_matrix(Y)
  

def least_squares_cg(Cui, X, Y, lambda_val, cg_steps=3):
    users, features = X.shape
    
    YtY = Y.T.dot(Y) + lambda_val * np.eye(features)

    for u in range(users):

        x = X[u]
        r = -YtY.dot(x)

        for i, confidence in nonzeros(Cui, u):
            r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]

        p = r.copy()
        rsold = r.dot(r)

        for it in range(cg_steps):
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x

alpha_val = 15
conf_data = (data_sparse * alpha_val).astype('double')
user_vecs, item_vecs = implicit_als_cg(conf_data, iterations=20, features=20)


#------------------------------
# FIND SIMILAR ITEMS
#------------------------------

# Let's find similar artists to Jay-Z. 
# Note that this ID might be different for you if you're using
# the full dataset or if you've sliced it somehow. 
item_id = 10277

# Get the item row for Jay-Z
item_vec = item_vecs[item_id].T

# Calculate the similarity score between Mr Carter and other artists
# and select the top 10 most similar.
scores = item_vecs.dot(item_vec).toarray().reshape(1,-1)[0]
top_10 = np.argsort(scores)[::-1][:10]

artists = []
artist_scores = []

# Get and print the actual artists names and scores
for idx in top_10:
    artists.append(item_lookup.artist.loc[item_lookup.artist_id == str(idx)].iloc[0])
    artist_scores.append(scores[idx])

similar = pd.DataFrame({'artist': artists, 'score': artist_scores})

print(similar)



# Let's say we want to recommend artists for user with ID 2023
user_id = 2023

#------------------------------
# GET ITEMS CONSUMED BY USER
#------------------------------

# Let's print out what the user has listened to
consumed_idx = data_sparse[user_id,:].nonzero()[1].astype(str)
consumed_items = item_lookup.loc[item_lookup.artist_id.isin(consumed_idx)]
print(consumed_items)


#------------------------------
# CREATE USER RECOMMENDATIONS
#------------------------------

def recommend(user_id, data_sparse, user_vecs, item_vecs, item_lookup, num_items=10):
    """Recommend items for a given user given a trained model
    
    Args:
        user_id (int): The id of the user we want to create recommendations for.
        
        data_sparse (csr_matrix): Our original training data.
        
        user_vecs (csr_matrix): The trained user x features vectors
        
        item_vecs (csr_matrix): The trained item x features vectors
        
        item_lookup (pandas.DataFrame): Used to map artist ids to artist names
        
        num_items (int): How many recommendations we want to return:
        
    Returns:
        recommendations (pandas.DataFrame): DataFrame with num_items artist names and scores
    
    """
  
    # Get all interactions by the user
    user_interactions = data_sparse[user_id,:].toarray()

    # We don't want to recommend items the user has consumed. So let's
    # set them all to 0 and the unknowns to 1.
    user_interactions = user_interactions.reshape(-1) + 1 #Reshape to turn into 1D array
    user_interactions[user_interactions > 1] = 0

    # This is where we calculate the recommendation by taking the 
    # dot-product of the user vectors with the item vectors.
    rec_vector = user_vecs[user_id,:].dot(item_vecs.T).toarray()

    # Let's scale our scores between 0 and 1 to make it all easier to interpret.
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_interactions*rec_vector_scaled
   
    # Get all the artist indices in order of recommendations (descending) and
    # select only the top "num_items" items. 
    item_idx = np.argsort(recommend_vector)[::-1][:num_items]

    artists = []
    scores = []

    # Loop through our recommended artist indicies and look up the actial artist name
    for idx in item_idx:
        artists.append(item_lookup.artist.loc[item_lookup.artist_id == str(idx)].iloc[0])
        scores.append(recommend_vector[idx])

    # Create a new dataframe with recommended artist names and scores
    recommendations = pd.DataFrame({'artist': artists, 'score': scores})
    
    return recommendations

# Let's generate and print our recommendations
recommendations = recommend(user_id, data_sparse, user_vecs, item_vecs, item_lookup)
print(recommendations)
