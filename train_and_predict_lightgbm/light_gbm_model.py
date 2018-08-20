import numpy as np
import pandas as pd
import lightgbm as lgb

print('Loading data...')
data_path = '../data/'

als = pd.read_csv(data_path + 'als_df.csv', dtype={'msno': 'category',
                                                   'source_system_tab': 'category',
                                                   'source_screen_name': 'category',
                                                   'source_type': 'category',
                                                   'target': np.uint8,
                                                   'song_id': 'category'})

train = pd.read_csv(data_path + 'train_df.csv', dtype={'msno': 'category',
                                                       'source_system_tab': 'category',
                                                       'source_screen_name': 'category',
                                                       'source_type': 'category',
                                                       'target': np.uint8,
                                                       'song_id': 'category'})

test = pd.read_csv(data_path + 'test_df.csv', dtype={'msno': 'category',
                                                     'source_system_tab': 'category',
                                                     'source_screen_name': 'category',
                                                     'source_type': 'category',
                                                     'target': np.uint8,
                                                     'song_id': 'category'})

valid = pd.read_csv(data_path + 'valid_df.csv', dtype={'msno': 'category',
                                                       'source_system_tab': 'category',
                                                       'source_screen_name': 'category',
                                                       'source_type': 'category',
                                                       'target': np.uint8,
                                                       'song_id': 'category'})

test_test = pd.read_csv(data_path + 'test.csv', dtype={'msno': 'category',
                                                  'source_system_tab': 'category',
                                                  'source_screen_name': 'category',
                                                  'source_type': 'category',
                                                  'song_id': 'category'})
songs = pd.read_csv(data_path + 'songs.csv', dtype={'genre_ids': 'category',
                                                    'language': 'category',
                                                    'artist_name': 'category',
                                                    'composer': 'category',
                                                    'lyricist': 'category',
                                                    'song_id': 'category'})
members = pd.read_csv(data_path + 'members.csv', dtype={'city': 'category',
                                                        'bd': np.uint8,
                                                        'gender': 'category',
                                                        'registered_via': 'category'})
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')


def create_statisctics_for_msno_df(df):
    msno2mean = df.groupby(['msno']).mean()['target'].to_dict()
    # msno2cnt = df.groupby(['msno']).count()['target'].to_dict()
    msno2cnt = df.groupby(['msno']).sum()['target'].to_dict()
    return msno2mean, msno2cnt


def create_statisctics_for_song_df(df):
    song2mean = df.groupby(['song_id']).mean()['target'].to_dict()
    # song2cnt = df.groupby(['song_id']).count()['target'].to_dict()
    song2cnt = df.groupby(['song_id']).sum()['target'].to_dict()
    return song2mean, song2cnt


def create_statistics_for_msno_source_system_tab(df):
    df_grp_by_smno_sst = df.groupby(['msno', 'source_system_tab']).mean()['target'].reset_index()
    df_grp_by_smno_sst = df_grp_by_smno_sst.dropna()
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['msno'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['source_system_tab'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'].astype('str')
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['new_index2'].astype('str')
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'] + df_grp_by_smno_sst['new_index2']
    df_grp_by_smno_sst.index = df_grp_by_smno_sst['new_index1'].values
    return df_grp_by_smno_sst['target'].to_dict()


def create_statistics_for_msno_source_screen_name(df):
    df_grp_by_smno_sst = df.groupby(['msno', 'source_screen_name']).mean()['target'].reset_index()
    df_grp_by_smno_sst = df_grp_by_smno_sst.dropna()
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['msno'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['source_screen_name'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'].astype('str')
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['new_index2'].astype('str')
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'] + df_grp_by_smno_sst['new_index2']
    df_grp_by_smno_sst.index = df_grp_by_smno_sst['new_index1'].values
    return df_grp_by_smno_sst['target'].to_dict()


def create_statistics_for_msno_source_type(df):
    df_grp_by_smno_sst = df.groupby(['msno', 'source_type']).mean()['target'].reset_index()
    df_grp_by_smno_sst = df_grp_by_smno_sst.dropna()
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['msno'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['source_type'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'].astype('str')
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['new_index2'].astype('str')
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'] + df_grp_by_smno_sst['new_index2']
    df_grp_by_smno_sst.index = df_grp_by_smno_sst['new_index1'].values
    return df_grp_by_smno_sst['target'].to_dict()


def create_statistics_for_song_source_system_tab(df):
    df_grp_by_smno_sst = df.groupby(['song_id', 'source_system_tab']).mean()['target'].reset_index()
    df_grp_by_smno_sst = df_grp_by_smno_sst.dropna()
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['song_id'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['source_system_tab'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'].astype('str')
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['new_index2'].astype('str')
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'] + df_grp_by_smno_sst['new_index2']
    df_grp_by_smno_sst.index = df_grp_by_smno_sst['new_index1'].values
    return df_grp_by_smno_sst['target'].to_dict()


def create_statistics_for_song_source_screen_name(df):
    df_grp_by_smno_sst = df.groupby(['song_id', 'source_screen_name']).mean()['target'].reset_index()
    df_grp_by_smno_sst = df_grp_by_smno_sst.dropna()
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['song_id'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['source_screen_name'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'].astype('str')
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['new_index2'].astype('str')
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'] + df_grp_by_smno_sst['new_index2']
    df_grp_by_smno_sst.index = df_grp_by_smno_sst['new_index1'].values
    return df_grp_by_smno_sst['target'].to_dict()


def create_statistics_for_song_source_type(df):
    df_grp_by_smno_sst = df.groupby(['song_id', 'source_type']).mean()['target'].reset_index()
    df_grp_by_smno_sst = df_grp_by_smno_sst.dropna()
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['song_id'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['source_type'].apply(lambda x: str(x))
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'].astype('str')
    df_grp_by_smno_sst['new_index2'] = df_grp_by_smno_sst['new_index2'].astype('str')
    df_grp_by_smno_sst['new_index1'] = df_grp_by_smno_sst['new_index1'] + df_grp_by_smno_sst['new_index2']
    df_grp_by_smno_sst.index = df_grp_by_smno_sst['new_index1'].values
    return df_grp_by_smno_sst['target'].to_dict()


def create_user_sst_mean_columns(df, dict_of_sst):
    cntr = 0
    sst_column = []
    for row in df[['msno', 'source_system_tab']].itertuples():
        msno = str(row[1])
        sst = str(row[2])
        key_for_sst = msno + sst
        if key_for_sst in dict_of_sst:
            sst_column.append(dict_of_sst[key_for_sst])
            cntr += 1
        else:
            sst_column.append(np.nan)
    print(cntr)
    return sst_column


def create_user_ssn_mean_columns(df, dict_of_sst):
    cntr = 0
    sst_column = []
    for row in df[['msno', 'source_screen_name']].itertuples():
        msno = str(row[1])
        sst = str(row[2])
        key_for_sst = msno + sst
        if key_for_sst in dict_of_sst:
            sst_column.append(dict_of_sst[key_for_sst])
            cntr += 1
        else:
            sst_column.append(np.nan)
    print(cntr)
    return sst_column


def create_user_st_mean_columns(df, dict_of_sst):
    cntr = 0
    sst_column = []
    for row in df[['msno', 'source_type']].itertuples():
        msno = str(row[1])
        sst = str(row[2])
        key_for_sst = msno + sst
        if key_for_sst in dict_of_sst:
            sst_column.append(dict_of_sst[key_for_sst])
            cntr += 1
        else:
            sst_column.append(np.nan)
    print(cntr)
    return sst_column


def create_song_sst_mean_columns(df, dict_of_sst):
    cntr = 0
    sst_column = []
    for row in df[['song_id', 'source_system_tab']].itertuples():
        msno = str(row[1])
        sst = str(row[2])
        key_for_sst = msno + sst
        if key_for_sst in dict_of_sst:
            sst_column.append(dict_of_sst[key_for_sst])
            cntr += 1
        else:
            sst_column.append(np.nan)
    print(cntr)
    return sst_column


def create_song_ssn_mean_columns(df, dict_of_sst):
    cntr = 0
    sst_column = []
    for row in df[['song_id', 'source_screen_name']].itertuples():
        msno = str(row[1])
        sst = str(row[2])
        key_for_sst = msno + sst
        if key_for_sst in dict_of_sst:
            sst_column.append(dict_of_sst[key_for_sst])
            cntr += 1
        else:
            sst_column.append(np.nan)
    print(cntr)
    return sst_column


def create_song_st_mean_columns(df, dict_of_sst):
    cntr = 0
    sst_column = []
    for row in df[['song_id', 'source_type']].itertuples():
        msno = str(row[1])
        sst = str(row[2])
        key_for_sst = msno + sst
        if key_for_sst in dict_of_sst:
            sst_column.append(dict_of_sst[key_for_sst])
            cntr += 1
        else:
            sst_column.append(np.nan)
    print(cntr)
    return sst_column

msno2mean_train, msno2cnt_train = create_statisctics_for_msno_df(als)
msno2mean_valid, msno2cnt_valid = create_statisctics_for_msno_df(pd.concat([als, train], axis=0))
msno2mean_test, msno2cnt_test = create_statisctics_for_msno_df(pd.concat([als, train, valid], axis=0))
msno2mean_test_test, msno2cnt_test_test = create_statisctics_for_msno_df(pd.concat([als, train, valid, test], axis=0))

song2mean_train, song2cnt_train = create_statisctics_for_song_df(als)
song2mean_valid, song2cnt_valid = create_statisctics_for_song_df(pd.concat([als, train], axis=0))
song2mean_test, song2cnt_test = create_statisctics_for_song_df(pd.concat([als, train, valid], axis=0))
song2mean_test_test, song2cnt_test_test = create_statisctics_for_song_df(pd.concat([als, train, valid, test], axis=0))

mson_sst_train = create_statistics_for_msno_source_system_tab(als)
mson_sst_valid = create_statistics_for_msno_source_system_tab(pd.concat([als, train], axis=0))
mson_sst_test = create_statistics_for_msno_source_system_tab(pd.concat([als, train, valid], axis=0))
mson_sst_test_test = create_statistics_for_msno_source_system_tab(pd.concat([als, train, valid, test], axis=0))

mson_ssn_train = create_statistics_for_msno_source_screen_name(als)
mson_ssn_valid = create_statistics_for_msno_source_screen_name(pd.concat([als, train], axis=0))
mson_ssn_test = create_statistics_for_msno_source_screen_name(pd.concat([als, train, valid], axis=0))
mson_ssn_test_test = create_statistics_for_msno_source_screen_name(pd.concat([als, train, valid, test], axis=0))

mson_st_train = create_statistics_for_msno_source_type(als)
mson_st_valid = create_statistics_for_msno_source_type(pd.concat([als, train], axis=0))
mson_st_test = create_statistics_for_msno_source_type(pd.concat([als, train, valid], axis=0))
mson_st_test_test = create_statistics_for_msno_source_type(pd.concat([als, train, valid, test], axis=0))

song_sst_train = create_statistics_for_song_source_system_tab(als)
song_sst_valid = create_statistics_for_song_source_system_tab(pd.concat([als, train], axis=0))
song_sst_test = create_statistics_for_song_source_system_tab(pd.concat([als, train, valid], axis=0))
song_sst_test_test = create_statistics_for_song_source_system_tab(pd.concat([als, train, valid, test], axis=0))

song_ssn_train = create_statistics_for_song_source_screen_name(als)
song_ssn_valid = create_statistics_for_song_source_screen_name(pd.concat([als, train], axis=0))
song_ssn_test = create_statistics_for_song_source_screen_name(pd.concat([als, train, valid], axis=0))
song_ssn_test_test = create_statistics_for_song_source_screen_name(pd.concat([als, train, valid, test], axis=0))

song_st_train = create_statistics_for_song_source_type(als)
song_st_valid = create_statistics_for_song_source_type(pd.concat([als, train], axis=0))
song_st_test = create_statistics_for_song_source_type(pd.concat([als, train, valid], axis=0))
song_st_test_test = create_statistics_for_song_source_type(pd.concat([als, train, valid, test], axis=0))

song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')
valid = valid.merge(songs[song_cols], on='song_id', how='left')
# test_test
test_test = test_test.merge(songs[song_cols], on='song_id', how='left')

members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan

songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis=1, inplace=True)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')
valid = valid.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on='song_id', how='left')
test = test.merge(songs_extra, on='song_id', how='left')
valid = valid.merge(songs_extra, on='song_id', how='left')


test_test = test_test.merge(members, on='msno', how='left')

test_test = test_test.merge(songs_extra, on='song_id', how='left')

import gc

del members, songs;
gc.collect();

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        valid[col] = valid[col].astype('category')

        test_test[col] = test_test[col].astype('category')


train['msno_mean'] = train['msno'].apply(lambda x: msno2mean_train[x] if x in msno2mean_train else np.nan)
valid['msno_mean'] = valid['msno'].apply(lambda x: msno2mean_valid[x] if x in msno2mean_valid else np.nan)
test['msno_mean'] = test['msno'].apply(lambda x: msno2mean_test[x] if x in msno2mean_test else np.nan)

train['song_mean'] = train['song_id'].apply(lambda x: song2mean_train[x] if x in song2mean_train else np.nan)
valid['song_mean'] = valid['song_id'].apply(lambda x: song2mean_valid[x] if x in song2mean_valid else np.nan)
test['song_mean'] = test['song_id'].apply(lambda x: song2mean_test[x] if x in song2mean_test else np.nan)

test_test['msno_mean'] = test_test['msno'].apply(lambda x: msno2mean_test_test[x] if x in msno2mean_test_test else np.nan)
test_test['song_mean'] = test_test['song_id'].apply(lambda x: song2mean_test_test[x] if x in song2mean_test_test else np.nan)

sst_mean_train = create_user_sst_mean_columns(train, mson_sst_train)
sst_mean_valid = create_user_sst_mean_columns(valid, mson_sst_valid)
sst_mean_test = create_user_sst_mean_columns(test, mson_sst_test)
sst_mean_test_test = create_user_sst_mean_columns(test_test, mson_sst_test_test)

train['user_sst_mean'] = sst_mean_train
valid['user_sst_mean'] = sst_mean_valid
test['user_sst_mean'] = sst_mean_test
test_test['user_sst_mean'] = sst_mean_test_test

ssn_mean_train = create_user_ssn_mean_columns(train, mson_ssn_train)
ssn_mean_valid = create_user_ssn_mean_columns(valid, mson_ssn_valid)
ssn_mean_test = create_user_ssn_mean_columns(test, mson_ssn_test)
ssn_mean_test_test = create_user_ssn_mean_columns(test_test, mson_ssn_test_test)

train['user_ssn_mean'] = ssn_mean_train
valid['user_ssn_mean'] = ssn_mean_valid
test['user_ssn_mean'] = ssn_mean_test
test_test['user_ssn_mean'] = ssn_mean_test_test

st_mean_train = create_user_st_mean_columns(train, mson_st_train)
st_mean_valid = create_user_st_mean_columns(valid, mson_st_valid)
st_mean_test = create_user_st_mean_columns(test, mson_st_test)
st_mean_test_test = create_user_st_mean_columns(test_test, mson_st_test_test)

train['user_st_mean'] = st_mean_train
valid['user_st_mean'] = st_mean_valid
test['user_st_mean'] = st_mean_test
test_test['user_st_mean'] = st_mean_test_test

sst_mean_train_sng = create_song_sst_mean_columns(train, song_sst_train)
sst_mean_valid_sng = create_song_sst_mean_columns(valid, song_sst_valid)
sst_mean_test_sng = create_song_sst_mean_columns(test, song_sst_test)
sst_mean_test_test_sng = create_song_sst_mean_columns(test_test, song_sst_test_test)

train['song_sst_mean'] = sst_mean_train_sng
valid['song_sst_mean'] = sst_mean_valid_sng
test['song_sst_mean'] = sst_mean_test_sng
test_test['song_sst_mean'] = sst_mean_test_test_sng

ssn_mean_train_sng = create_song_ssn_mean_columns(train, song_ssn_train)
ssn_mean_valid_sng = create_song_ssn_mean_columns(valid, song_ssn_valid)
ssn_mean_test_sng = create_song_ssn_mean_columns(test, song_ssn_test)
ssn_mean_test_test_sng = create_song_ssn_mean_columns(test_test, song_ssn_test_test)

train['song_ssn_mean'] = ssn_mean_train_sng
valid['song_ssn_mean'] = ssn_mean_valid_sng
test['song_ssn_mean'] = ssn_mean_test_sng
test_test['song_ssn_mean'] = ssn_mean_test_test_sng

st_mean_train_sng = create_song_st_mean_columns(train, song_st_train)
st_mean_valid_sng = create_song_st_mean_columns(valid, song_st_valid)
st_mean_test_sng = create_song_st_mean_columns(test, song_st_test)
st_mean_test_test_sng = create_song_st_mean_columns(test_test, song_st_test_test)

train['song_st_mean'] = st_mean_train_sng
valid['song_st_mean'] = st_mean_valid_sng
test['song_st_mean'] = st_mean_test_sng
test_test['song_st_mean'] = st_mean_test_test_sng

del train['song_id']
del valid['song_id']
del test['song_id']
del test_test['song_id']
del train['msno']
del valid['msno']
del test['msno']
del test_test['msno']

del train['artist_name']
del valid['artist_name']
del test['artist_name']
del test_test['artist_name']

train_artists = pd.read_csv('../train_als_vectors/train_with_artist_score.csv')
valid_artists = pd.read_csv('../train_als_vectors/valid_with_artist_score.csv')
test_artists = pd.read_csv('../train_als_vectors/test_with_artist_score.csv')
test_test_artists = pd.read_csv('../train_als_vectors/test_test_with_artist_score.csv')

train_songs = pd.read_csv('../train_als_vectors/train_with_song_score.csv')
valid_songs = pd.read_csv('../train_als_vectors/valid_with_song_score.csv')
test_songs = pd.read_csv('../train_als_vectors/test_with_song_score.csv')
test_test_songs = pd.read_csv('../train_als_vectors/test_test_with_song_score.csv')

new_columns = ['als_score']
train = pd.concat([train, train_artists[new_columns]], axis=1)
valid = pd.concat([valid, valid_artists[new_columns]], axis=1)
test = pd.concat([test, test_artists[new_columns]], axis=1)
test_test = pd.concat([test_test, test_test_artists[new_columns]], axis=1)

columns = list(train.columns)
columns[-1] = 'als_score_artist'
train.columns = columns
valid.columns = columns
test.columns = columns
columns = list(test_test.columns)
columns[-1] = 'als_score_artist'
test_test.columns = columns

train = pd.concat([train, train_songs[new_columns]], axis=1)
valid = pd.concat([valid, valid_songs[new_columns]], axis=1)
test = pd.concat([test, test_songs[new_columns]], axis=1)
test_test = pd.concat([test_test, test_test_songs[new_columns]], axis=1)

train['als_score_artist'] = train['als_score_artist'].fillna(train['als_score_artist'].mean())
valid['als_score_artist'] = valid['als_score_artist'].fillna(valid['als_score_artist'].mean())
test['als_score_artist'] = test['als_score_artist'].fillna(test['als_score_artist'].mean())
test_test['als_score_artist'] = test_test['als_score_artist'].fillna(test_test['als_score_artist'].mean())

train = pd.concat([train, valid, test], axis=0)
test = test_test

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

print(train.describe())
print(test.describe())
print(train.info())

X = train.drop(['target'], axis=1)
y = train['target'].values

X_test = test.drop(['id'], axis=1)
ids = test['id'].values

del train, test
gc.collect()

d_train = lgb.Dataset(X, y)
watchlist = [d_train]

print('Training LGBM model...')
params = dict()
params['learning_rate'] = 0.2
params['application'] = 'binary'
params['max_depth'] = 8
params['num_leaves'] = 2 ** 8
params['verbosity'] = 0
params['metric'] = 'auc'

model = lgb.train(params, train_set=d_train, num_boost_round=50, valid_sets=watchlist, verbose_eval=5)

print('Making predictions and saving them...')
p_test = model.predict(X_test)

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv('submission_mean_msno_song_id_als_artists_songs_minus_artists_sst_ssn_st_mean_sst_ssn_st_song.csv.gz',
            compression='gzip', index=False, float_format='%.5f')
print('Done!')
