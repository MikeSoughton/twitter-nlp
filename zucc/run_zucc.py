import os
import json

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

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

        """
        
        """

        self.tweets_df = pd.read_csv(tweets_file_path)
        
        


    def prepare_data(self, tokenizer_file_path, max_len):
        print("Preparing data...")

        with open(tokenizer_file_path) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)

        # Get the sentances into a list
        list_sentences = self.tweets_df["Text"].fillna("Invalid").values

        #tokenizer = text.Tokenizer(num_words=max_features)
        #tokenizer.fit_on_texts(list(list_sentences))

        # Get the data into padded sequences
        list_tokenized = tokenizer.texts_to_sequences(list_sentences)
        self.x = sequence.pad_sequences(list_tokenized, maxlen=max_len)

        # Because the dataset is massive and takes forever to run the whole thing, reduce if specified
        #self.reduced_data_size = reduced_data_size
        #if reduced_data_size is not None:
        #    self.x_train = self.x_train[:reduced_data_size]
        #    self.y_train = self.y_train[:reduced_data_size]
        #    self.x_test = self.x_test[:int(reduced_data_size*test_size)]
        #    self.y_test = self.y_test[:int(reduced_data_size*test_size)]

        print(self.x.shape)


    def build_model(self, model_name, max_features, max_len):
        print("Building model...")

        self.model = get_model(model_name, conv_layers = 2, max_dilation_rate = 4, max_features = max_features, max_len = max_len)
        self.model.summary()
        

    def run_model(self, model_dir, model_name, data_used_name, trained_weights_path, batch_size = 512):

        file_path = model_dir + "checkpoints/" + model_name + "/" + data_used_name + "/" + trained_weights_path
        self.model.load_weights(file_path)
        print("Loaded model weights from", file_path)

        #y_pred = self.model.predict(self.x, batch_size=batch_size, verbose=True)
        #y_pred_rounded = np.round(y_pred)

        #y_pred_decoded = y_pred[:,1]

        #self.tweets_df['Bot scores'] = y_pred_decoded

        #self.tweets_df.to_csv('tttt.csv')
        tweets_df = pd.read_csv('tttt.csv')


        pd.set_option("display.max_rows", None, "display.max_columns", None)

        print(tweets_df['Text'].loc[np.where((tweets_df['Bot scores'] > 0.75))[0]])

        print(len(tweets_df['Text'].loc[np.where((tweets_df['Bot scores'] > 0.75))[0]]))
        print(len(tweets_df['Text']))



        #print(y_pred)
            

    def test_model(self, batch_size = 1024):
        print("Testing model...")
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix

        y_pred = self.model.predict(self.x_test, batch_size=batch_size, verbose=True)
        y_pred_rounded = np.round(y_pred)

        y_test_decoded = np.argmax(self.y_test, axis=-1)
        print("Accuracy:", accuracy_score(self.y_test, y_pred_rounded))

        y_pred_rounded_decoded = np.argmax(y_pred_rounded, axis=-1)
        species = np.array(y_test_decoded)
        predictions = np.array(y_pred_rounded_decoded)
        confusion_matrix = confusion_matrix(species, predictions)

        print("Confusion matrix:")
        print(confusion_matrix)

