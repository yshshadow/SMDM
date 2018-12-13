import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import collections
import os
import pickle


class TwitterCluster:
    """
    get k-means result of tweets data
    """
    def __init__(self):
        self.commons_path = './model/KMeans_commons.pickle'
        self.cluster_path = './model/KMeans_model.joblib'
        if os.path.isfile(self.cluster_path) and os.path.isfile(self.commons_path):
            self.cluster = joblib.load(self.cluster_path)
            with open(self.commons_path, 'rb') as f:
                self.commons = pickle.load(f)
        else:
            self.cluster, self.commons = self.train_cluster()

    def train_cluster(self, save_model=True):
        """
        train k-means model
        :param save_model: if True, save model to file
        :return: cluster: k-means model
                 commons: 500 common words
        """
        train_df = self.load_training_data()

        # use nltk TweetTokenizer to tokenize tweets
        tt = TweetTokenizer()
        # some punctuation are not included in string.punctuation, remove them too
        # tokenize every single tweet
        train_df['tokens'] = train_df.apply(lambda df: tt.tokenize(df['text']), axis=1)
        stop_words = set(stopwords.words('english')
                         + list(string.punctuation)
                         + ['...', '“', '’', '”', '‘', '–']
                         + ['u'])
        freq_dist = self.get_freq_dist(train_df, stop_words)
        # use 500 common words
        commons = sorted(freq_dist.items(), key=lambda kv: len(kv[1]), reverse=True)[:500]

        # get vectorized sparse matrix
        matrix = self.get_matrix(train_df, commons)

        # cluster
        cluster = KMeans(10, n_jobs=-1)
        cluster.fit(matrix)
        if save_model:
            joblib.dump(cluster, self.cluster_path)
            with open(self.commons_path, 'wb') as f:
                pickle.dump(commons, f)
        return cluster, commons

    def load_training_data(self):
        """
        load training data from csv
        :return: training data
        """
        # read tweets data from csv file
        df = pd.read_csv('./data/Tweets.csv', header=0)
        # change all words in tweet text to lower case
        df['text'] = df['text'].str.lower()
        return df

    def get_freq_dist(self, df, stop_words):
        """
        get frequency distribution
        :param df: input data
        :param stop_words: words need to be removed
        :return: frequency distribution dictionary
        """
        freqdist = collections.defaultdict(lambda: [])
        for i in range(len(df)):
            tokens = df.loc[i, 'tokens']
            for token in tokens:
                # skip numbers, @ string, word in stop list and words have quotation
                if token.isdigit() or token in stop_words or token.startswith('@'):
                    continue
                freqdist[token].append(i)
        return freqdist

    def get_matrix(self, df, commons):
        """
        get vectorized matrix
        :param df: input dataframe
        :param commons: common words
        :return: a matrix
        """
        array = np.zeros((len(df), len(commons)), dtype=np.int32)
        for idx, item in enumerate(commons):
            for line in item[1]:
                array[line, idx] = 1
        return pd.DataFrame(array, dtype=np.int32, columns=[key for key, v in commons])

    def most_representative(self, size=5):
        """
        get the representative words of clusters
        :param size: the number of representative words
        :return: dataframe with representative words
        """
        representative = pd.DataFrame(index=range(0, 10), columns=range(1, size + 1))
        for index, array in enumerate(self.cluster.cluster_centers_):
            r_index = np.argsort(array)[::-1][:size]
            for pos, value in enumerate(r_index):
                representative.iloc[index, pos] = self.commons[value][0]
        return representative

    def get_labels(self):
        return self.cluster.labels_
