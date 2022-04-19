import glob
import os
import re

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Tensorflow literally always has warnings!

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout, concatenate
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from models import build_model

class ZuccDestroyer():
    def __init__(self, max_features, max_len):

        print("Hello")
        self.max_features = max_features

    def read_data(self, human_tweets_dir, bot_tweets_dir, human_tweets_files = None, bot_tweets_files = None, reduced_data_size = None):
        print("Reading in data...")


        russian_troll_tweets = None
        if bot_tweets_files is None:
            IRA_files = glob.glob(bot_tweets_dir + "*.csv")
            print(IRA_files)
            
            li = []

            for filename in sorted(IRA_files):
                print(filename)
                df = pd.read_csv(filename, index_col=None, header=0)
                li.append(df)

            russian_troll_tweets = pd.concat(li, axis=0, ignore_index=True)

        elif bot_tweets_files is not None and len(bot_tweets_files) > 1:
            russian_troll_tweets = None
            for filename in sorted(os.listdir(bot_tweets_dir)):
                if filename in bot_tweets_files:
                    print(filename)
                    russian_troll_tweets_i = pd.read_csv(bot_tweets_dir + filename)
                    if russian_troll_tweets is not None:
                        russian_troll_tweets = pd.concat([russian_troll_tweets, russian_troll_tweets_i])
                    else:
                        russian_troll_tweets = russian_troll_tweets_i
                        
        elif bot_tweets_files is not None and len(bot_tweets_files) == 1:
            russian_troll_tweets = pd.read_csv(bot_tweets_dir + bot_tweets_files[0])

        self.russian_troll_tweets = russian_troll_tweets #TODO is it more efficient to define this above?

        print(self.russian_troll_tweets.shape)
    
        # Human tweets 
        #TODO Can make this dynamic as above
        self.human_tweets = pd.read_csv(human_tweets_dir + human_tweets_files) 
        print(self.human_tweets)

    def prepare_data(self, max_features, max_len):

        # Get the sentances into a list
        list_sentences_bots = self.russian_troll_tweets["content"].fillna("Invalid").values
        list_sentences_humans = self.human_tweets["Text"].fillna("Invalid").values
        list_sentences_train = np.concatenate((list_sentences_humans, list_sentences_bots), axis = 0)

        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(list_sentences_train))

        # Get the data into padded sequences
        list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
        x = sequence.pad_sequences(list_tokenized_train, maxlen=max_len)

        y_humans = np.zeros((len(list_sentences_humans)))
        y_bots = np.ones((len(list_sentences_bots)))
        y = np.concatenate((y_humans, y_bots), axis = 0)
        y = to_categorical(y)


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        print(x_train, y_train)

    def build_model2(self, model_name, max_features, max_len):

        conv_layers = 2
        max_dilation_rate = 4
        model = build_model(model_name, conv_layers = conv_layers, max_dilation_rate = max_dilation_rate, max_features = max_features, max_len = max_len)
        model.summary()
        
        return model

