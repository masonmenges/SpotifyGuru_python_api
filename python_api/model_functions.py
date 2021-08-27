import numpy as np
import joblib
import pandas as pd

nn = joblib.load('./python_api/knn.gz')
df = pd.read_csv('./python_api/song_names.csv')


def suggest_songs(song_data, number=10):
    song_ids = list()

    song_array = np.array(
        [song_data['stats']['acousticness'],
         song_data['stats']['danceability'],
         song_data['stats']['energy'],
         song_data['stats']['instrumentalness'],
         song_data['stats']['key'],
         song_data['stats']['liveness'],
         song_data['stats']['loudness'],
         song_data['stats']['mode'],
         song_data['stats']['speechiness'],
         song_data['stats']['tempo'],
         song_data['stats']['valence'],
         song_data['year']]
    )

    dist, songs_index = nn.kneighbors(
        song_array.reshape(1, -1), n_neighbors=number)

    for val in songs_index.tolist()[0]:
        song_id = df['id'][val]
        song_ids.append(song_id)

    return song_ids
