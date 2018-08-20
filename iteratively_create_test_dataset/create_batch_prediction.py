import pandas as pd
import pickle
import numpy as np
import lightgbm as lgb
import time

from sklearn.metrics import roc_auc_score

index_msno = 1
index_target = -1
index_artist_name = 8
index_song_id = 2
index_source_system_tab = 3
index_source_type = 5
index_source_screen_name = 4
index_genre_ids = 7

df = pd.read_csv('/srv/reporting/kkbox/train.csv')

df = df.iloc[-1475484:].copy()

df_songs_extra = pd.read_csv('/srv/reporting/kkbox/new_df_songs.csv')
df_songs = pd.read_csv('/srv/reporting/kkbox/songs.csv')
df_members = pd.read_csv('/srv/reporting/kkbox/members.csv')

df = pd.merge(df, df_songs, on='song_id', how='left')
df = pd.merge(df, df_songs_extra, on='song_id', how='left')
df = pd.merge(df, df_members, on='msno', how='left')

df = df.head(10000)

model = lgb.Booster(model_file='/srv/reporting/kkbox/model.model')
# model = pickle.load(open('/srv/reporting/light_gbm_model.p', 'rb'))

full_target = []

with open('/srv/reporting/kkbox/big_dict_train.p', 'rb') as f:
# with open('/srv/reporting/kkbox/big_dict_py2.p', 'rb') as f:
    big_dict = pickle.load(f)


def get_from_dict(value_type, key):
    if key in big_dict[value_type].keys():
        mean_d, count_d = big_dict[value_type][key]
    else:
        mean_d, count_d = 0, 0
    return mean_d, count_d


def add_to_dict(value_type, key, target):
    if key in big_dict[value_type].keys():
        mean_d, count_d = big_dict[value_type][key]
        big_dict[value_type][key] = ((mean_d * count_d + target)/float(count_d + 1), count_d + 1)
    else:
        mean_d, count_d = 0, 0
        big_dict[value_type][key] = (0.5, 1)
    return mean_d, count_d


def update_dict(df):
    for row in df.itertuples():
        tmp_m, tmp_c = add_to_dict("msno", row[index_msno], row[index_target])
        tmp_m, tmp_c = add_to_dict("artist", row[index_artist_name], row[index_target]) 
        tmp_m, tmp_c = add_to_dict("song_id", row[index_song_id], row[index_target])
        tmp_m, tmp_c = add_to_dict("sst", row[index_source_system_tab], row[index_target])
        tmp_m, tmp_c = add_to_dict("st", row[index_source_type], row[index_target])
        tmp_m, tmp_c = add_to_dict("ssn", row[index_source_screen_name], row[index_target])
        tmp_m, tmp_c = add_to_dict("user_ssn", row[index_msno] + str(row[index_source_screen_name]), row[index_target])
        tmp_m, tmp_c = add_to_dict("user_sst", row[index_msno] + str(row[index_source_system_tab]), row[index_target])
        tmp_m, tmp_c = add_to_dict("user_st", row[index_msno] + str(row[index_source_type]), row[index_target])
        tmp_m, tmp_c = add_to_dict("song_ssn", row[index_song_id] + str(row[index_source_screen_name]), row[index_target])
        tmp_m, tmp_c = add_to_dict("song_sst", row[index_song_id] + str(row[index_source_system_tab]), row[index_target])
        tmp_m, tmp_c = add_to_dict("song_st", row[index_song_id] + str(row[index_source_type]), row[index_target])
        tmp_m, tmp_c = add_to_dict("user_artist", row[index_msno] + str(row[index_artist_name]), row[index_target])
        tmp_m, tmp_c = add_to_dict("user_genre", row[index_msno] + str(row[index_genre_ids]), row[index_target])
        tmp_m, tmp_c = add_to_dict("genre", str(row[index_genre_ids]), row[index_target])


def create_df_to_predict_mode(df_batch):
    t0 = time.time()
    y = df_batch['target'].values
    df_batch = df_batch.drop(['target'], axis=1)
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
    for row in df_batch.itertuples():
        tmp_m, tmp_c = get_from_dict("msno", row[index_msno])
        msno_cnt.append(tmp_c)
        msno_mean.append(tmp_m)
        tmp_m, tmp_c = get_from_dict("artist", row[index_artist_name])
        artist_cnt.append(tmp_c)
        artist_mean.append(tmp_m)
        tmp_m, tmp_c = get_from_dict("song_id", row[index_song_id])
        song_cnt.append(tmp_c)
        song_mean.append(tmp_m)
        tmp_m, tmp_c = get_from_dict("sst", row[index_source_system_tab])
        sst_mean.append(tmp_m)
        tmp_m, tmp_c = get_from_dict("st", row[index_source_type])
        st_mean.append(tmp_m)
        tmp_m, tmp_c = get_from_dict("ssn", row[index_source_screen_name])
        ssn_mean.append(tmp_m)
        tmp_m, tmp_c = get_from_dict("user_ssn", row[index_msno] + str(row[index_source_screen_name]))
        user_ssn_mean.append(tmp_m)
        user_ssn_count.append(tmp_c)
        tmp_m, tmp_c = get_from_dict("user_sst", row[index_msno] + str(row[index_source_system_tab]))
        user_sst_mean.append(tmp_m)
        user_sst_count.append(tmp_c)
        tmp_m, tmp_c = get_from_dict("user_st", row[index_msno] + str(row[index_source_type]))
        user_st_mean.append(tmp_m)
        user_st_count.append(tmp_c)
        tmp_m, tmp_c = get_from_dict("song_ssn", row[index_song_id] + str(row[index_source_screen_name]))
        song_ssn_mean.append(tmp_m)
        song_ssn_count.append(tmp_c)
        tmp_m, tmp_c = get_from_dict("song_sst", row[index_song_id] + str(row[index_source_system_tab]))
        song_sst_mean.append(tmp_m)
        song_sst_count.append(tmp_c)
        tmp_m, tmp_c = get_from_dict("song_st", row[index_song_id] + str(row[index_source_type]))
        song_st_mean.append(tmp_m)
        song_st_count.append(tmp_c)
        tmp_m, tmp_c = get_from_dict("user_artist", row[index_msno] + str(row[index_artist_name]))
        user_artist_mean.append(tmp_m)
        user_artist_count.append(tmp_c)
        tmp_m, tmp_c = get_from_dict("user_genre", row[index_msno] + str(row[index_genre_ids]))
        user_genre_mean.append(tmp_m)
        user_genre_count.append(tmp_c)
        tmp_m, tmp_c = get_from_dict("genre", str(row[index_genre_ids]))
        genre_mean.append(tmp_m)
        genre_count.append(tmp_c)
    # print('create columns', time.time() - t0)
    # t0 = time.time()

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
    # print('create dataframe', time.time() - t0)
    # t0 = time.time()
    target = model.predict(new_df)
    # print('predict', time.time() - t0)
    # t0 = time.time()
    df_batch['target'] = target
    update_dict(df_batch)
    # print('update dict', time.time() - t0)
    # t0 = time.time()
    full_target.extend(target)
    # print('extend full target', time.time() - t0)
    # t0 = time.time()
    # print('roc_auc', roc_auc_score(y, target))


batch_size = 1
number_of_batches = int(df.shape[0] / batch_size)

for i in range(number_of_batches + 1):
# for i in range(1):
    if i % 100 == 0:
        print('batch_id', i)
    df_batch = df.iloc[i*batch_size: (i+1)*batch_size, :].copy()
    create_df_to_predict_mode(df_batch)

with open('/srv/reporting/kkbox/big_dict_updated.p', 'wb') as f:
    pickle.dump(big_dict, f)

pickle.dump(full_target, open('/srv/reporting/kkbox/test_target.pickle', 'wb'), protocol=2)
print(roc_auc_score(df['target'], full_target))
