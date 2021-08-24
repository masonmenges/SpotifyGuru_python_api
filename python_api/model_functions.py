import demjson
import numpy as np
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.models import Model
import joblib
import os
from sklearn.neighbors import NearestNeighbors
import pandas as pd


song = """{
  spotifyID     : '6RUhbFEhrvGISaQ8u2j2JN',
  title         : 'Dangerous Woman',
  author        : 'Ariana Grande',
  album         : 'Dangerous Woman',
  albumImageLink: 'https://i.scdn.co/image/ab67616d0000b273628d506d5bddb09099db242c',
  audioLink     : 'https://api.spotify.com/v1/tracks/6RUhbFEhrvGISaQ8u2j2JN',
  year          : '2016',
  stats         : {
    acousticness    : 0.0529,
    danceability    : 0.664,
    energy          : 0.602,
    instrumentalness: 0,
    key             : 4,
    liveness        : 0.356,
    loudness        : -5.369,
    mode            : 0,
    speechiness     : 0.0412,
    tempo           : 134.049,
    valence         : 0.289
  }
}"""

test = demjson.decode(song)


test_array = np.array(
    [test['stats']['acousticness'],
    test['stats']['danceability'],
    test['stats']['energy'],
    test['stats']['instrumentalness'],
    test['stats']['key'],
    test['stats']['liveness'],
    test['stats']['loudness'],
    test['stats']['mode'],
    test['stats']['speechiness'],
    test['stats']['tempo'],
    test['stats']['valence'],
    test['year']]
)

# inp = Input(shape=(12,))
#
# enc1 = Dense(6, activation='relu')(inp)
# enc2 = Dense(1, activation='relu')(enc1)
#
# enc = Model(inp, enc2)

# encoded_song = enc.predict(test_array.reshape(1,-1))

nn = joblib.load("./knn.gz")

dist, songs_index = nn.kneighbors(test_array.reshape(1,-1))

df = pd.read_csv('./song_names.csv')

song_ids = list()

for val in songs_index.tolist()[0]:
    song_id = df['id'][val]
    song_ids.append(song_id)
print(song_id)

print(song_ids)
