import glob
import os
import io
import json

import numpy as np 
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Tensorflow literally always has warnings!

from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from models import get_model

class TrainEurekaDroid():
    def __init__(self):
        pass

    def read_data(self, filter_on_english, human_tweets_dir, bot_tweets_dir, human_tweets_files = None, bot_tweets_files = None):
        print("Reading in data...")

        """
        Here we initially read in either:
            1) All files in the bot dataset folder
            2) Multiple specific files in the bot dataset folder
            3) A single specific file in the bot dataset folder

        Next we read in the human dataset file. Currently this is a single
        file but in the future it may consist of multiple files and could
        be read in in a similar fashion.
        """

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

        # Human tweets 
        #TODO Can make this dynamic as above if applicable
        self.human_tweets = pd.read_csv(human_tweets_dir + human_tweets_files) 

        # If there are NaN value problems filter them out
        self.russian_troll_tweets = self.russian_troll_tweets.dropna(subset=self.russian_troll_tweets.select_dtypes(float).columns, how='all')        
        self.russian_troll_tweets.reset_index(drop=True, inplace=True)

        self.human_tweets = self.human_tweets.dropna(subset=['Text'], how='all')
        self.human_tweets.reset_index(drop=True, inplace=True)

        # Filtering English tweets
        print(self.russian_troll_tweets.shape, self.human_tweets.shape)
        if filter_on_english is True:
            self.russian_troll_tweets = russian_troll_tweets.drop(self.russian_troll_tweets[~self.russian_troll_tweets['language'].str.contains('English')].index)
            #TODO investigate why I need na=False here since no NaNs are found in the human dataset
            self.human_tweets = self.human_tweets.drop(self.human_tweets[~self.human_tweets['Language'].str.contains('en', na=False)].index)
        print(self.russian_troll_tweets.shape, self.human_tweets.shape)


    def prepare_data(self, max_features, max_len, test_size, reduced_data_size = None):
        print("Preparing data...")

        # Get the sentances into a list
        list_sentences_bots = self.russian_troll_tweets["content"].fillna("Invalid").values
        list_sentences_humans = self.human_tweets["Text"].fillna("Invalid").values
        list_sentences_train = np.concatenate((list_sentences_humans, list_sentences_bots), axis = 0)

        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(list_sentences_train))

        # We will save the tokenizer for later use so make it self here to be saved when we save the model
        self.tokenizer = tokenizer

        # Get the data into padded sequences
        list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
        x = sequence.pad_sequences(list_tokenized_train, maxlen=max_len)

        y_humans = np.zeros((len(list_sentences_humans)))
        y_bots = np.ones((len(list_sentences_bots)))
        y = np.concatenate((y_humans, y_bots), axis = 0)
        y = to_categorical(y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = test_size, random_state=42)

        # Because the dataset is massive and takes forever to run the whole thing, reduce if specified
        self.reduced_data_size = reduced_data_size
        if reduced_data_size is not None:
            self.x_train = self.x_train[:reduced_data_size]
            self.y_train = self.y_train[:reduced_data_size]
            self.x_test = self.x_test[:int(reduced_data_size*test_size)]
            self.y_test = self.y_test[:int(reduced_data_size*test_size)]


    def build_model(self, model_name, max_features, max_len):
        print("Building model...")

        self.model = get_model(model_name, conv_layers = 2, max_dilation_rate = 4, max_features = max_features, max_len = max_len)
        self.model.summary()
        

    def train_model(self, model_dir, model_name, data_used_name, epochs, batch_size, load_weights, pretrained_weights_path = None):

        if load_weights is True:
            file_path = model_dir + "checkpoints/" + model_name + "/" + data_used_name + "/" + pretrained_weights_path
            self.model.load_weights(file_path)
            print("Loaded model weights from", file_path)
            
        else:
            print("Training model...")

            # Where the weights will be saved to:
            if not os.path.exists(model_dir + "checkpoints/" + model_name):
                os.makedirs(model_dir + "checkpoints/" + model_name)
            
            if self.reduced_data_size is not None:
                file_path = model_dir + "checkpoints/" + model_name + "/" + data_used_name + "/{epoch:02d}epochs_" + str(batch_size) + "batch_" + str(self.reduced_data_size) + "reduceddata_weights.hdf5"
            else:
                file_path = model_dir + "checkpoints/" + model_name + "/" + data_used_name + "/{epoch:02d}epochs_" + str(batch_size) + "batch_fulldata_weights.hdf5"

            print(file_path)
            checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

            callbacks_list = [checkpoint, early]
            self.model.fit(self.x_train, self.y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_split=0.1, 
                    callbacks=callbacks_list)
            #model.load_weights(file_path)

        # Save the tokenizer here. We could have saved it earlier on but now we can save it in the same
        # path as the model. Note that pickle is more efficient than json but json is nicer and more reliable
        tokenizer_json = self.tokenizer.to_json()
        tokenizer_file_path = model_dir + "checkpoints/" + model_name + "/" + data_used_name + "/tokenizer.json"
        with io.open(tokenizer_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

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

