import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import TweetTokenizer
import pickle
import os


class TwitterSentimentAnalyzer:
    def __init__(self):
        self.model_path = './model/SA_model.pickle'
        self.words_path = './model/W_features.pickle'
        self.tknzr = TweetTokenizer()
        self.clf, self.word_features = self.load_sentiment_model()

    def predict(self, features):
        result = self.clf.classify(features)
        return 'pos' if result == 4 else 'neg'

    def predict_df(self, text_df):
        text_df['tokens'] = text_df.apply(lambda row: self.tknzr.tokenize(row['text']), axis=1)
        # text_df = self.tokenize(text_df)
        predict_set = [self.document_features(d, self.word_features) for i, d in text_df.iterrows()]
        senti_result = pd.DataFrame(text_df['id'], columns=['id', 'sentiment'])
        for i in senti_result.index:
            senti_result.loc[i, 'sentiment'] = self.predict(predict_set[i])
        return senti_result

    def load_sentiment_model(self):
        if not os.path.isfile(self.model_path) or not os.path.isfile(self.words_path):
            return self.train_sentiment_model()
        with open(self.model_path, 'rb') as f:
            clf = pickle.load(f)
        with open(self.words_path, 'rb') as f:
            word_features = pickle.load(f)
        return clf, word_features

    def train_sentiment_model(self, save_model=True):
        # use data from sentiment140, see http://help.sentiment140.com/for-students
        train_df = self.load_training_data('./data/training.1600000.processed.noemoticon.csv')
        # tokenize text
        # train_df['tokens'] = train_df.apply(lambda row: self.tknzr.tokenize(row['text']), axis=1)
        train_df = self.tokenize(train_df)
        # stop words, includes nltk stop words, punctuations
        stop_words = set(stopwords.words('english')
                         + list(string.punctuation)
                         + ['..', '...', '“', '’', '”', '‘', '–'])
        all_words = self.get_freq_dist(train_df, stop_words)
        word_features = [w for (w, c) in all_words.most_common(500)]

        # training data used in nltk classifier
        train_set = [(self.document_features(d, word_features), d['sentiment']) for i, d in train_df.iterrows()]
        clf = nltk.NaiveBayesClassifier.train(train_set)
        if save_model:
            with open(self.model_path, 'wb') as f:
                pickle.dump(clf, f)
            with open(self.words_path, 'wb') as f:
                pickle.dump(word_features, f)
        return clf, word_features

    def load_training_data(self, path, encoding='iso-8859-1', sample_size=40000):
        # Import data from sentiment140 csv file
        train_df = pd.read_csv(path, encoding=encoding, header=None)
        # remove useless columns
        train_df = train_df.loc[:, [0, 5]]
        train_df.columns = ['sentiment', 'text']
        # sample in negative data and positive data
        neg_train_df = train_df[train_df['sentiment'] == 0]
        pos_train_df = train_df[train_df['sentiment'] == 4]
        neg_train_df = neg_train_df.sample(n=sample_size // 2)
        pos_train_df = pos_train_df.sample(n=sample_size // 2)
        train_df = pd.concat([neg_train_df, pos_train_df])
        train_df['text'] = train_df['text'].str.lower()

        return train_df

    def get_freq_dist(self, df, stop_words):
        freqdist = nltk.FreqDist()
        for i in df.index:
            tokens = df.loc[i, 'tokens']
            for token in tokens:
                # skip numbers, @ string, word in stop words set
                if token.isdigit() or token in stop_words or token.startswith('@'):
                    continue
                freqdist[token] += 1
        return freqdist

    def tokenize(self, df):
        # tknzr = TweetTokenizer()
        # def func(row):
        #     self.tknzr.tokenize(row['text'])
        #
        # df['tokens'] = df.apply(func, axis=1)
        df['tokens'] = df.apply(lambda row: self.tknzr.tokenize(row['text']), axis=1)
        return df

    def document_features(self, document, word_features):
        doc_words = set(document['tokens'])
        doc_features = {}
        for word in word_features:
            doc_features['contains({})'.format(word)] = (word in doc_words)
        # print(features)
        return doc_features

# sa = TwitterSentimentAnalyzer()
# comments = pd.read_csv('./data/Comments.csv')

