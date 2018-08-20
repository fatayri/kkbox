import numpy as np
import pandas as pd
import lightgbm as lgb
import implicit
import gc

from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, accuracy_score


def create_rating_matrix_for_user_song(df):
    codes_of_user = dict()
    codes_of_item = dict()
    free_code_for_user = 0
    free_code_for_item = 0

    item_codes = []
    user_codes = []
    ratings = []

    for row in df.itertuples():
        user_id = row[1]
        song_id = row[2]
        rating = row[3]

        if user_id in codes_of_user:
            user_code = codes_of_user[user_id]
        else:
            codes_of_user[user_id] = free_code_for_user
            user_code = free_code_for_user
            free_code_for_user += 1

        if song_id in codes_of_item:
            item_code = codes_of_item[song_id]
        else:
            codes_of_item[song_id] = free_code_for_item
            item_code = free_code_for_item
            free_code_for_item += 1

        item_codes.append(item_code)
        user_codes.append(user_code)
        if rating == 0:
            ratings.append(0.1)
        else:
            ratings.append(1.0)
    return item_codes, user_codes, ratings, codes_of_user, codes_of_item

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

test_test = pd.read_csv(data_path + 'test.csv', dtype={'msno': 'category',
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

songs = pd.read_csv(data_path + 'songs.csv', dtype={'genre_ids': 'category',
                                                    'language': 'category',
                                                    'artist_name': 'category',
                                                    'composer': 'category',
                                                    'lyricist': 'category',
                                                    'song_id': 'category'})

als = pd.merge(als, songs[['song_id', 'artist_name']], on='song_id', how='left')
train = pd.merge(train, songs[['song_id', 'artist_name']], on='song_id', how='left')
valid = pd.merge(valid, songs[['song_id', 'artist_name']], on='song_id', how='left')
test = pd.merge(test, songs[['song_id', 'artist_name']], on='song_id', how='left')
test_test = pd.merge(test_test, songs[['song_id', 'artist_name']], on='song_id', how='left')

col_names = ['msno', 'artist_name', 'target']
als = als[col_names]
train = train[col_names]
valid = valid[col_names]
test = test[col_names]
test_test = test_test[['msno', 'artist_name']]

FACTORS = 40
REGUL = 0.01

item_codes, user_codes, ratings, codes_of_user_als, codes_of_item_als = create_rating_matrix_for_user_song(als)
ratings_matrix = csr_matrix((ratings, (item_codes, user_codes)), shape=(len(codes_of_item_als), len(codes_of_user_als)))
alpha = 40
model = implicit.als.AlternatingLeastSquares(factors=FACTORS, num_threads=2, regularization=REGUL,
                                             calculate_training_loss=True)
model.fit(ratings_matrix * alpha)
songs_factors_als = model.item_factors
users_factors_als = model.user_factors

als_pred_train = []

for row in train.itertuples():
    user_id = row[1]
    song_id = row[2]
    if user_id in codes_of_user_als and song_id in codes_of_item_als:
        als_pred_train.append(np.dot(songs_factors_als[codes_of_item_als[song_id]], users_factors_als[codes_of_user_als[user_id]]))
    else:
        als_pred_train.append(np.nan)


item_codes, user_codes, ratings, codes_of_user_als_train, codes_of_item_als_train = create_rating_matrix_for_user_song(pd.concat([als, train]))
ratings_matrix = csr_matrix((ratings, (item_codes, user_codes)), shape=(len(codes_of_item_als_train), len(codes_of_user_als_train)))
alpha = 40
model = implicit.als.AlternatingLeastSquares(factors=FACTORS, num_threads=2, regularization=REGUL,
                                             calculate_training_loss=True)
model.fit(ratings_matrix * alpha)
songs_factors_als_train = model.item_factors
users_factors_als_train = model.user_factors

als_pred_valid = []

for row in valid.itertuples():
    user_id = row[1]
    song_id = row[2]
    if user_id in codes_of_user_als_train and song_id in codes_of_item_als_train:
        als_pred_valid.append(np.dot(songs_factors_als_train[codes_of_item_als_train[song_id]], users_factors_als_train[codes_of_user_als_train[user_id]]))
    else:
        als_pred_valid.append(np.nan)


item_codes, user_codes, ratings, codes_of_user_als_train_valid, codes_of_item_als_train_valid = create_rating_matrix_for_user_song(pd.concat([als, train, valid]))
ratings_matrix = csr_matrix((ratings, (item_codes, user_codes)), shape=(len(codes_of_item_als_train_valid), len(codes_of_user_als_train_valid)))
alpha = 40
model = implicit.als.AlternatingLeastSquares(factors=FACTORS, num_threads=2, regularization=REGUL,
                                             calculate_training_loss=True)
model.fit(ratings_matrix * alpha)
songs_factors_als_train_valid = model.item_factors
users_factors_als_train_valid = model.user_factors

als_pred_test = []

for row in test.itertuples():
    user_id = row[1]
    song_id = row[2]
    if user_id in codes_of_user_als_train_valid and song_id in codes_of_item_als_train_valid:
        als_pred_test.append(np.dot(songs_factors_als_train_valid[codes_of_item_als_train_valid[song_id]], users_factors_als_train_valid[codes_of_user_als_train_valid[user_id]]))
    else:
        als_pred_test.append(np.nan)

item_codes, user_codes, ratings, codes_of_user_als_train_valid, codes_of_item_als_train_valid = create_rating_matrix_for_user_song(pd.concat([als, train, valid, test]))
ratings_matrix = csr_matrix((ratings, (item_codes, user_codes)), shape=(len(codes_of_item_als_train_valid), len(codes_of_user_als_train_valid)))
alpha = 40
model = implicit.als.AlternatingLeastSquares(factors=FACTORS, num_threads=2, regularization=REGUL,
                                             calculate_training_loss=True)
model.fit(ratings_matrix * alpha)
songs_factors_als_train_valid = model.item_factors
users_factors_als_train_valid = model.user_factors

als_pred_test_test = []

for row in test_test.itertuples():
    user_id = row[1]
    song_id = row[2]
    if user_id in codes_of_user_als_train_valid and song_id in codes_of_item_als_train_valid:
        als_pred_test_test.append(np.dot(songs_factors_als_train_valid[codes_of_item_als_train_valid[song_id]], users_factors_als_train_valid[codes_of_user_als_train_valid[user_id]]))
    else:
        als_pred_test_test.append(np.nan)

train_new = train.copy()
valid_new = valid.copy()
test_new = test.copy()
test_test_new = test_test.copy()

train_new['als_score'] = als_pred_train
valid_new['als_score'] = als_pred_valid
test_new['als_score'] = als_pred_test
test_test_new['als_score'] = als_pred_test_test

train_new.to_csv('train_with_artist_score.csv', index=0)
valid_new.to_csv('valid_with_artist_score.csv', index=0)
test_new.to_csv('test_with_artist_score.csv', index=0)
test_test_new.to_csv('test_test_with_artist_score.csv', index=0)

train_new = train_new[['als_score', 'target']]
valid_new = valid_new[['als_score', 'target']]
test_new = test_new[['als_score', 'target']]

train_new = pd.concat([train_new, valid_new])

train_mean = train_new['als_score'].mean()
train_new['als_score'] = train_new['als_score'].fillna(train_mean)
test_new['als_score'] = test_new['als_score'].fillna(train_mean)

X_train, y_train = train_new[['als_score']].values, train_new['target'].values
X_test, y_test = test_new[['als_score']].values, test_new['target'].values

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

print('train_valid_test', roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
print('test', roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# train_valid_test 0.582790050547
# test 0.560902366335
