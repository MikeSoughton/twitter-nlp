import os
import json

import numpy as np 
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Tensorflow literally always has warnings!

from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_preprocessing.text import tokenizer_from_json


from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from models import get_model

class RunZucc():
    def __init__(self):
        pass

    def read_data(self, tweets_file_path):
        print("Reading in data...")

        self.tweets_file_path = tweets_file_path
        self.tweets_df = pd.read_csv(tweets_file_path)

    def prepare_data(self, tokenizer_file_path, max_len):
        print("Preparing data...")

        # Load the tokenizer that was saved in training
        with open(tokenizer_file_path) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)

        # Get the sentances into a list
        list_sentences = self.tweets_df["Text"].fillna("Invalid").values

        # Get the data into padded sequences
        list_tokenized = tokenizer.texts_to_sequences(list_sentences)
        self.x = sequence.pad_sequences(list_tokenized, maxlen=max_len)

        print(self.x.shape)


    def build_model(self, model_name, max_features, max_len):
        print("Building model...")

        self.model = get_model(model_name, conv_layers = 2, max_dilation_rate = 4, max_features = max_features, max_len = max_len)
        self.model.summary()
        

    def run_model(self, model_dir, model_name, data_used_name, trained_weights_path, data_out_dir, batch_size = 512):

        weights_file_path = model_dir + "checkpoints/" + model_name + "/" + data_used_name + "/" + trained_weights_path
        self.model.load_weights(weights_file_path)
        print("Loaded model weights from", weights_file_path)

        # Run classifier over data
        y_pred = self.model.predict(self.x, batch_size=batch_size, verbose=True)

        # Unpacking the one-hot encoded predictions
        y_pred_decoded = y_pred[:,1]

        # Save dataframe
        self.tweets_df['Bot scores'] = y_pred_decoded
        #tweets_df = pd.read_csv('tttt.csv')

        #TODO assumes data saved in data/scraped_tweets/<keywords>/
        keywords_dir = self.tweets_file_path.split("/")[2]
        data_file_name = self.tweets_file_path.split("/")[3]
        if not os.path.exists(data_out_dir + keywords_dir):
            os.makedirs(data_out_dir + keywords_dir)
        
        self.tweets_df.to_csv(data_out_dir + keywords_dir + "/" + data_file_name.replace('.csv', '_classi.csv'))
        print("Saved tweets with classifier bot scores to", str(data_out_dir + keywords_dir + "/" + data_file_name.replace('.csv', '_classi.csv')))


        # My experimenting - delete after it is saved elsewhere:
        """
        tweets_df = pd.read_csv('tttt.csv')
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        #print(tweets_df['Text'].loc[np.where((tweets_df['Bot scores'] > 0.75))[0]])

        #print(len(tweets_df['Text'].loc[np.where((tweets_df['Bot scores'] > 0.75))[0]]))
        #print(len(tweets_df['Text']))

        #print(tweets_df['Datetime'])
        #print(tweets_df[tweets_df['Datetime'].str.contains("2022-02-24", na=False)])
        #print(tweets_df['Datetime'].split("-"))
        start_date_idx = np.where(tweets_df['Datetime'].str.contains("2022-02-24", na=False))[0][0]
        print("star", start_date_idx)

        
        before_df = tweets_df.loc[:(start_date_idx - 2)] # idk why - 2 and not - 1 but it works
        before_df = before_df.reset_index(drop=True)

        after_df = tweets_df.loc[start_date_idx:]
        after_df = after_df.reset_index(drop=True)
        #print(before_df['Text'].loc[np.where((before_df['Bot scores'] > 0.75))[0]])
        #print(after_df['Text'].loc[np.where((after_df['Bot scores'] > 0.75))[0]])
        #print(tweets_df.loc[start_date:, 'Datetime'])

        print(len(before_df['Text'].loc[np.where((before_df['Bot scores'] > 0.75))[0]]))
        print(len(after_df['Text'].loc[np.where((after_df['Bot scores'] > 0.75))[0]]))

        #print(before_df['Text'])
        #print(after_df['Text'])

        print(len(before_df['Text']))
        print(len(after_df['Text']))

        #print(after_df[['Text', 'Bot scores']])
        """
        
            

