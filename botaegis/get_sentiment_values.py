from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import re
#import textblob           
#from textblob import TextBlob

import numpy as np 
import pandas as pd


class GetSentiment:
    def __init__(self):
        pass

    def read_data(self, tweets_file_path):
        print("Reading in data...")
        print(tweets_file_path)

        self.tweets_file_path = tweets_file_path
        self.tweets_df = pd.read_csv(tweets_file_path)

        # Drop any NaN values if they exist
        self.tweets_df = self.tweets_df.dropna(subset=['Username'], how='all')
        #self.tweets_df = self.tweets_df.dropna(subset=['Text'], how='all')
        self.tweets_df = self.tweets_df.reset_index(drop=True)

    def prepare_data(self):
        """
        Performs a number of operations to clean the data. These are
        """

        print("Preparing data...")

        # Making lowercase
        #self.tweets_df['Text_lower'] = self.tweets_df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        self.tweets_df['Text_lower'] = self.tweets_df['Text'].str.lower()

        # Removing punctuation
        self.tweets_df['Text_punc'] = self.tweets_df['Text'].str.replace('[^\w\s]','')

        # Removing stopwords
        nltk.download('stopwords') #This must be run once to download the stopwords, but nltk knows if it exists already
        stop = stopwords.words('english')

        self.tweets_df['Text_stop']  = self.tweets_df['Text_punc'] .apply(lambda x: " ".join(x for x in x.split() if x not in stop))

        # Tokenising tweets
        def tokenization(text):
            text = re.split('\W+', text)
            return text
        self.tweets_df['Text_tokenized'] = self.tweets_df['Text_stop'].apply(lambda x: tokenization(x.lower()))

        # Lemmatising the tweets and put the tokenised text back together
        nltk.download('wordnet') #This must be run once to download the wordnet, but nltk knows if it exists already
        nltk.download('omw-1.4') # Also must be installed
        wn = nltk.WordNetLemmatizer()
        def lemmatizer(text):
            text = ' '.join([wn.lemmatize(word) for word in text])
            return text

        self.tweets_df['Text_lemmatized'] = self.tweets_df['Text_tokenized'].apply(lambda x: lemmatizer(x))
        self.tweets_df[['Text', 'Text_punc', 'Text_tokenized','Text_stop','Text_lemmatized']][0:90]

        drop_intermediate_cols = True
        if drop_intermediate_cols is True:
            self.tweets_df = self.tweets_df.drop(columns=['Text_lower', 'Text_punc', 'Text_tokenized', 'Text_stop'])

        print(self.tweets_df.head())

    def run_sentiment(self):
        print("Obtaining sentiment scores...")

        nltk.download('vader_lexicon') #This must be run once to download the wordnet, but nltk knows if it exists already
        analyser = SentimentIntensityAnalyzer()

        scores=[]
        for i in range(len(self.tweets_df['Text_lemmatized'])):
            score = analyser.polarity_scores(self.tweets_df['Text_lemmatized'][i])
            print(i, self.tweets_df['Text_lemmatized'][i], score)
            score = score['compound']
            scores.append(score)
            
            
        self.tweets_df['score']= pd.Series(np.array(scores))

        print(np.mean(scores))

        print(self.tweets_df.head())

