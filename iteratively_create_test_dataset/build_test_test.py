# coding: utf-8

import pandas as pd
import pickle
import numpy as np

df = pd.read_csv('/srv/reporting/kkbox/test.csv')
model = pickle.load('/srv/reporting/light_gbm_model.p')

df_songs_extra = pd.read_csv('/srv/reporting/kkbox/new_df_songs.csv')
df_songs = pd.read_csv('/srv/reporting/kkbox/songs.csv')
df_members = pd.read_csv('/srv/reporting/kkbox/members.csv')

df = pd.merge(df, df_songs, on='song_id', how='left')
df = pd.merge(df, df_songs_extra, on='song_id', how='left')
df = pd.merge(df, df_members, on='msno', how='left')
df.head()

big_dict = pickle.load(open('big_dict.p', 'rb'))

small_dict = {}

for key in big_dict.keys():
    small_dict[key] = np.mean([x[0] for x in big_dict[key].values()])


def add_to_dict(value_type, key):
    if key in big_dict[value_type].keys():
        mean_d, count_d = big_dict[value_type][key]
        big_dict[value_type][key] = (mean_d, count_d + 1)
    else:
        mean_d, count_d = 0, 0
        big_dict[value_type][key] = (small_dict[value_type], 1)
    return mean_d, count_d

msno_cnt = []
msno_mean = []
artist_cnt = []
artist_mean = []
song_cnt = []
song_mean = []
genre_mean = []
genre_count = []
user_genre_mean = []
user_genre_count = []
sst_mean = []
st_mean = []
ssn_mean = []
user_sst_mean = []
user_ssn_mean = []
user_st_mean = []
song_sst_mean = []
song_ssn_mean = []
song_st_mean = []
user_sst_count = []
user_ssn_count = []
user_st_count = []
song_sst_count = []
song_ssn_count = []
song_st_count = []
user_artist_mean = []
user_artist_count = []

for index, row in df.iterrows():
    tmp_m, tmp_c = add_to_dict("msno", row["msno"])
    msno_cnt.append(tmp_c)
    msno_mean.append(tmp_m)
    tmp_m, tmp_c = add_to_dict("artist", row["artist_name"])
    artist_cnt.append(tmp_c)
    artist_mean.append(tmp_m)
    tmp_m, tmp_c = add_to_dict("song_id", row["song_id"])
    song_cnt.append(tmp_c)
    song_mean.append(tmp_m)
    tmp_m, tmp_c = add_to_dict("sst", row["source_system_tab"])
    sst_mean.append(tmp_m)
    tmp_m, tmp_c = add_to_dict("st", row["source_type"])
    st_mean.append(tmp_m)
    tmp_m, tmp_c = add_to_dict("ssn", row["source_screen_name"])
    ssn_mean.append(tmp_m)
    tmp_m, tmp_c = add_to_dict("user_ssn", row["msno"] + str(row["source_screen_name"]))
    user_ssn_mean.append(tmp_m)
    user_ssn_count.append(tmp_c)
    tmp_m, tmp_c = add_to_dict("user_sst", row["msno"] + str(row["source_system_tab"]))
    user_sst_mean.append(tmp_m)
    user_sst_count.append(tmp_c)
    tmp_m, tmp_c = add_to_dict("user_st", row["msno"] + str(row["source_type"]))
    user_st_mean.append(tmp_m)
    user_st_count.append(tmp_c)
    tmp_m, tmp_c = add_to_dict("song_ssn", row["song_id"] + str(row["source_screen_name"]))
    song_ssn_mean.append(tmp_m)
    song_ssn_count.append(tmp_c)
    tmp_m, tmp_c = add_to_dict("song_sst", row["song_id"] + str(row["source_system_tab"]))
    song_sst_mean.append(tmp_m)
    song_sst_count.append(tmp_c)
    tmp_m, tmp_c = add_to_dict("song_st", row["song_id"] + str(row["source_type"]))
    song_st_mean.append(tmp_m)
    song_st_count.append(tmp_c)
    tmp_m, tmp_c = add_to_dict("user_artist", row["msno"] + str(row["artist_name"]))
    user_artist_mean.append(tmp_m)
    user_artist_count.append(tmp_c)
    tmp_m, tmp_c = add_to_dict("user_genre", row["msno"] + str(row["genre_ids"]))
    user_genre_mean.append(tmp_m)
    user_genre_count.append(tmp_c)
    tmp_m, tmp_c = add_to_dict("genre", str(row["genre_ids"]))
    genre_mean.append(tmp_m)
    genre_count.append(tmp_c)

    if index % 10000 == 0:
        print(index)

new_df = pd.DataFrame({'msno_cnt': msno_cnt,
                       'msno_mean': msno_mean,
                       'artist_cnt': artist_cnt,
                       'artist_mean': artist_mean,
                       'song_cnt': song_cnt,
                       'song_mean': song_mean,
                       'sst_mean': sst_mean,
                       'st_mean': st_mean,
                       'ssn_mean': ssn_mean,
                       'user_sst_mean': user_sst_mean,
                       'user_ssn_mean': user_ssn_mean,
                       'user_st_mean': user_st_mean,
                       'song_sst_mean': song_sst_mean,
                       'song_ssn_mean': song_ssn_mean,
                       'song_st_mean': song_st_mean,
                       'user_sst_count': user_sst_count,
                       'user_ssn_count': user_ssn_count,
                       'user_st_count': user_st_count,
                       'song_sst_count': song_sst_count,
                       'song_ssn_count': song_ssn_count,
                       'song_st_count': song_st_count,
                       'user_artist_mean': user_artist_mean,
                       'user_artist_count': user_artist_count,
                       'user_genre_mean': user_genre_mean,
                       'user_genre_count': user_genre_count,
                       'genre_mean': genre_mean,
                       'genre_count': genre_count})

new_df.to_csv('test_test_for_subm.csv', index=None)
