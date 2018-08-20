import random
import pandas as pd

SIZE_OF_TRAIN = 7377418

size_of_train_df = int(SIZE_OF_TRAIN * 0.6)
size_of_valid_df = (SIZE_OF_TRAIN - size_of_train_df) / 2
size_of_train_df /= 2
size_of_als_df = size_of_train_df


def divide_train_into_train_valid_and_test():
    with open('../data/train.csv', 'r') as train_main, open('../data/train_df.csv', 'w') as train,\
            open('../data/valid_df.csv', 'w') as valid, open('../data/test_df.csv', 'w') as test,\
            open('../data/als_df.csv', 'w') as als:
        first_line = next(train_main)
        als.write(first_line)
        train.write(first_line)
        valid.write(first_line)
        test.write(first_line)
        cntr = 0
        for line in train_main:
            if cntr < size_of_als_df:
                als.write(line)
                cntr += 1
            elif size_of_als_df <= cntr < (size_of_als_df + size_of_train_df):
                train.write(line)
                cntr += 1
            elif (size_of_als_df + size_of_train_df) <= cntr < (size_of_als_df + size_of_train_df + size_of_valid_df):
                valid.write(line)
                cntr += 1
            else:
                test.write(line)

if __name__ == '__main__':
    divide_train_into_train_valid_and_test()

    als = pd.read_csv('../data/als_df.csv')
    train = pd.read_csv('../data/train_df.csv')
    valid = pd.read_csv('../data/valid_df.csv')
    test = pd.read_csv('../data/test_df.csv')

    print(als['target'].sum() / als.shape[0])
    print(train['target'].sum() / train.shape[0])
    print(valid['target'].sum() / valid.shape[0])
    print(test['target'].sum() / test.shape[0])
