import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm
tqdm.pandas()
os.getcwd()
artists = pd.read_csv('data/artists.csv', index_col=0)
df = pd.read_csv('data/main_dataset.csv', index_col=0)
import ast
import itertools
def flatten_list(input_list):
    return list(itertools.chain(*input_list))

def concat_flatlist(input_list):
    input_list = [i.replace(' ','') for i in input_list]
    return ' '.join(input_list)

df['artists_genres'] = df['artists_genres'].apply(ast.literal_eval)
df['artists_genres_flat'] = df['artists_genres'].apply(flatten_list).apply(concat_flatlist)

from sklearn.model_selection import train_test_split
df['playlist_uris'] = df['playlist_uris'].apply(ast.literal_eval)
playlists = df['playlist_uris'].explode()
playlists = list(set(playlists))

playlists_songs = pd.DataFrame(index= playlists, columns = ['songs'])

def get_playlist_songs(playlist):
    return df[df['playlist_uris'].apply(lambda x: playlist in x)].index.tolist()

playlists_songs['songs'] = playlists_songs.progress_apply(lambda x: get_playlist_songs(x.name), axis=1)

playlists_songs  = playlists_songs[playlists_songs.index!='']

# Assuming playlist_songs is a list of songs
train_songs, test_songs = train_test_split(playlists_songs, test_size=0.2, random_state=42)

# train_songs will contain 80% of the playlist_songs
# test_songs will contain the remaining 20% of the playlist_songs

## Feature extraction
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

song_vectorizer = CountVectorizer()
song_vectorizer.fit(df['artists_genres_flat'])
song_vectorized = song_vectorizer.transform(df['artists_genres_flat'])


svd = TruncatedSVD(n_components=10)
song_vectorized_svd = svd.fit_transform(song_vectorized)
song_vectorized = None
from sklearn.decomposition import PCA

df_features = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

pca = PCA(n_components=5)
df_features_reduced = pca.fit_transform(df_features)

df['artists_names']  = df['artists_names'].apply(ast.literal_eval)
df['artists_uris']  = df['artists_uris'].apply(ast.literal_eval)

names = df['artists_names'].explode()
uris = df['artists_uris'].explode()

names_with_uris = pd.DataFrame({'names': names, 'uris': uris})
names_with_uris.set_index('uris'   , inplace=True)
names_with_uris.drop_duplicates(inplace=True)


names = list(set(names))


uris = list(set(uris))

import json

def get_artist_embeddings(artist_uris):
    df_filter = df_features_reduced[df['artists_uris'].apply(lambda x: artist_uris in x)]
    if df_filter.shape[0] == 0:
        return np.zeros((5,)).tolist()
    return df_filter.mean(axis=0).tolist()

generate_embeddings_artist = False 
if generate_embeddings_artist: 
    artist_embeddings = artists.progress_apply(lambda x: get_artist_embeddings(x.name),axis=1)
    artist_embeddings.to_csv('data/artist_embeddings.csv')

artist_embeddings = pd.read_csv('data/artist_embeddings.csv', index_col=0)
artist_embeddings.index = artist_embeddings.index.values


artist_embeddings = artist_embeddings.apply(lambda x: json.loads(x.iloc[0]), axis=1)

names_with_uris.index = names_with_uris.index.values
artist_embeddings_plot = artist_embeddings.progress_apply(lambda x: pd.Series(x))

track_artist_embeddings = df['artists_uris'].progress_apply(lambda x: artist_embeddings_plot.loc[x].mean())
track_artist_embeddings = track_artist_embeddings[[0,1,2]]

genres = df['artists_genres'].explode().explode()
genres = list(set(genres))
df['artists_genres_flattened_list'] = df['artists_genres'].apply(flatten_list)
genres_embedding = pd.DataFrame(index= genres, columns = range(5))
def get_genre_embeddings(genre):
    df_filter = df_features_reduced[df['artists_genres_flattened_list'].apply(lambda x: genre in x)]
    if df_filter.shape[0] == 0:
        return np.zeros((5,)).tolist()
    return df_filter.mean(axis=0).tolist()
generate_embeddings_genres = False 
if generate_embeddings_genres: 
    genres_embedding = genres_embedding.progress_apply(lambda x: get_genre_embeddings(x.name) ,axis=1)
    genres_embedding.to_csv('data/genre_embeddings.csv')
genres_embedding = pd.read_csv('data/genre_embeddings.csv', index_col=0)
genres_embedding.index = genres_embedding.index.values
genres_embedding = genres_embedding.apply(lambda x: json.loads(x.iloc[0]), axis=1)
genres_embedding = genres_embedding.progress_apply(lambda x: pd.Series(x))
track_genre_embeddings = df['artists_genres_flattened_list'].progress_apply(lambda x: genres_embedding.loc[x].mean())
track_genre_embeddings = track_genre_embeddings[[0,1,2]]

embeddings_with_artist_genre = track_genre_embeddings.merge(track_artist_embeddings,left_index = True, right_index = True)
final_embedding = pd.merge(df_features, embeddings_with_artist_genre.fillna(0), left_index = True, right_index = True)
final_embedding.to_csv('data/final_embedding.csv')
from annoy import AnnoyIndex
def build_ann(df, filename, dist_metric='angular', fit=True):
    f = df.shape[1]  
    if fit:

        t = AnnoyIndex(f, dist_metric)
        for i in range(df.shape[0]):
            t.add_item(i, df.iloc[i].values)
        t.build(30) # 10 trees
        t.save(filename)
        return t
    else:
        u = AnnoyIndex(f, dist_metric)
        
        u.load(filename)
        return u

u_cb = build_ann(final_embedding, 'models/cb_annoy.ann', dist_metric='angular', fit=True)

## Traditional Collaborative filtering
# Create an empty DataFrame with columns as unique song IDs
one_hot_vectors = pd.DataFrame(columns=df.index, index=train_songs.index)

def get_one_hot_vector(songs):
    one_hot_vector = pd.Series(0, index=df.index)
    one_hot_vector.loc[songs] = 1
    return one_hot_vector 

one_hot_vectors = one_hot_vectors.progress_apply(lambda x: get_one_hot_vector(train_songs.loc[x.name]['songs']), axis=1)



svd_playlists_test = TruncatedSVD(n_components=200)
svd_playlists_test.fit(one_hot_vectors)
pd.Series(svd_playlists_test.explained_variance_ratio_.cumsum(),index=range(1,201)).plot()
# Create an empty DataFrame with columns as unique song IDs
one_hot_vectors = pd.DataFrame(columns=df.index, index=train_songs.index)

def get_one_hot_vector(songs):
    one_hot_vector = pd.Series(0, index=df.index)
    one_hot_vector.loc[songs] = 1
    return one_hot_vector 

one_hot_vectors = one_hot_vectors.progress_apply(lambda x: get_one_hot_vector(train_songs.loc[x.name]['songs']), axis=1)


svd_playlists = TruncatedSVD(n_components=50)
one_hot_vectors_svd = svd_playlists.fit_transform(one_hot_vectors)
one_hot_vectors = None
import pickle
pickle.dump(svd_playlists, open('models/svd_playlists.pkl', 'wb'))
pd.DataFrame(one_hot_vectors_svd, index=train_songs.index).to_csv('data/playlist_svd.csv')
u_cf = build_ann(pd.DataFrame(one_hot_vectors_svd), 'models/cf_annoy.ann', dist_metric='angular', fit=True)
