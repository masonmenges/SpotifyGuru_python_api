import numpy as np
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.keras.models import load_model

config = ConfigProto(device_count={"GPU": 0})
session = Session(config=config)

with tf.device('/cpu:0'):
    model = load_model('./python_api/nn_model', compile=False)
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
         song_data['stats']['valence']]
    )
    with tf.device('/cpu:0'):
        pred = model.predict(song_array.reshape(1, -1))

    knn_input = np.array([pred[0][0], song_data['year']]).reshape(1, -1)

    dist, songs_index = nn.kneighbors(
        knn_input, n_neighbors=number)

    for val in songs_index.tolist()[0]:
        song_id = df['id'][val]
        song_ids.append(song_id)

    return song_ids
