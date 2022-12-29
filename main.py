import streamlit as st

st.markdown("# Sistem Rekomendasi Smartphone for Photograpy 🎈")
st.sidebar.markdown("# Photograpy 🎈")



# Data processing

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity

# Similarity

df = pd.read_csv("DATASET.csv")
df.index = np.arange(1, len(df)+1)
df.index.name = "ID"





del df ['Timestamp']
del df ['NAMA']
del df ['UMUR']
##df.head()

df.info()

# Create user-item matrix
matrix = df


# Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 'rows')


# User similarity matrix using Pearson correlation
user_similarity = matrix_norm.T.corr()


# Pick a user ID
picked_userid = int(st.number_input("Pilih User ID: "))



# Remove picked user ID from the candidate list
user_similarity.drop(index=picked_userid, inplace=True)

# Take a look at the data
user_similarity.head()

# Number of similar users
n = 10

# User similarity threashold
user_similarity_threshold = 0.35

# Get top n similar users
similar_users = user_similarity[user_similarity[picked_userid]>user_similarity_threshold][picked_userid].sort_values(ascending=False)[:n]

# Print out top n similar users
st.write(f'The similar users for user {picked_userid} are', similar_users)

# Movies that the target user has watched
picked_userid_watched = matrix_norm[matrix_norm.index == picked_userid].dropna(axis=1, how='all')
##picked_userid_watched

# Movies that similar users watched. Remove movies that none of the similar users have watched
similar_user_movies = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')
##similar_user_movies

# A dictionary to store item scores
item_score = {}

# Loop through items
for i in similar_user_movies.columns:
  # Get the ratings for movie i
  movie_rating = similar_user_movies[i]
  # Create a variable to store the score
  total = 0
  # Create a variable to store the number of scores
  count = 0
  # Loop through similar users
  for u in similar_users.index:
    # If the movie has rating
    if pd.isna(movie_rating[u]) == False:
      # Score is the sum of user similarity score multiply by the movie rating
      score = similar_users[u] * movie_rating[u]
      # Add the score to the total score for the movie so far
      total += score
      # Add 1 to the count
      count +=1
  # Get the average score for the item
  item_score[i] = total / count

# Convert dictionary to pandas dataframe
item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])
    
# Sort the movies by score
ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)

# Select top m movies
m = 5

# Average rating for the picked user
avg_rating = matrix[matrix.index == picked_userid].T.mean()[picked_userid]

# Print the average movie rating for user 1
st.write(f'The average movie rating for user {picked_userid} is {avg_rating:.2f}')

# Calcuate the predicted rating
ranked_item_score['predicted_rating'] = ranked_item_score['movie_score'] + avg_rating

# Take a look at the data
st.write(ranked_item_score.head(m))