import os 
import re
import numpy as np 
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class GetSentiment:
    def __init__(self, tweets_file_path):
        self.tweets_file_path = tweets_file_path

    def read_data(self):
        print("Reading in data...")
        print(self.tweets_file_path)

        self.tweets_df = pd.read_csv(self.tweets_file_path)

        # Drop any NaN values if they exist
        self.tweets_df = self.tweets_df.dropna(subset=['Username'], how='all')
        self.tweets_df = self.tweets_df.dropna(subset=['Text'], how='all')
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
        self.stop = stopwords.words('english')

        #self.tweets_df = self.tweets_df.loc[self.tweets_df.Text.apply(type) != float] # Would drop any floats, but all floats were NaNs which were dropped above
        self.tweets_df['Text_stop']  = self.tweets_df['Text_punc'] .apply(lambda x: " ".join(x for x in x.split() if x not in self.stop))

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

    def run_sentiment(self, data_out_dir, tweets_out_file_name_desc):
        print("Obtaining sentiment scores...")

        nltk.download('vader_lexicon') #This must be run once to download the wordnet, but nltk knows if it exists already
        analyser = SentimentIntensityAnalyzer()

        scores=[]
        for i in range(len(self.tweets_df['Text_lemmatized'])):
            score = analyser.polarity_scores(self.tweets_df['Text_lemmatized'][i])
            print(i, self.tweets_df['Text_lemmatized'][i], score)
            score = score['compound']
            scores.append(score)
            
            
        self.tweets_df['Sentiment score']= pd.Series(np.array(scores))

        print(np.mean(scores))

        print(self.tweets_df.head())

        # Setup directories for saving outputs:
        #TODO assumes data saved in data/scraped_tweets/<keywords>/
        keywords_dir = self.tweets_file_path.split("/")[2]

        if not os.path.exists(data_out_dir + keywords_dir):
            os.makedirs(data_out_dir + keywords_dir)

        self.tweets_df.to_csv(data_out_dir + keywords_dir + "/" + tweets_out_file_name_desc + '.csv')
        print("Saved tweets with classifier bot scores to", str(data_out_dir + keywords_dir + "/" + tweets_out_file_name_desc + '.csv'))


class LDA_Decomposition:
    def __init__(self, LDA_tweets_in_file_path):
        self.LDA_tweets_in_file_path = LDA_tweets_in_file_path

    def read_data(self):

        # We'll read in the data file saved from the sentiment part. This means that the sentiment part
        # must be run at least once before running this.
        print("Reading in data...")
        print(self.LDA_tweets_in_file_path)

        self.tweets_df = pd.read_csv(self.LDA_tweets_in_file_path)

        # Drop any NaN values if they exist
        self.tweets_df = self.tweets_df.dropna(subset=['Username'], how='all')
        self.tweets_df = self.tweets_df.dropna(subset=['Text'], how='all')
        self.tweets_df = self.tweets_df.dropna(subset=['Text_lemmatized'], how='all')
        self.tweets_df = self.tweets_df.reset_index(drop=True)

    def filter_stopwords(self):
        print("Filtering stopwords...")

        # DO WE WANT TO DO THIS? THINK ABOUT IT?

        nltk.download('stopwords') #This must be run once to download the stopwords, but nltk knows if it exists already
        stop = stopwords.words('english')
        
        # Add specific words to be filtered out. I think the rationale here is that since the subject 
        # is e.g. Ukraine, we aren't interest in learning that tweets concern Ukraine
        #new = ("ukraine", "russia")
        #stop.extend(new)

        print(stop[-10:])

        self.tweets_df['Tokens']  = self.tweets_df['Text_lemmatized'] .apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        self.tweets_df.head()

    def setup_countvectorizer(self, threshold, labelled_data, bot_classifier = 'botaegis'):
        print("Initialising CountVectorizer...")

        if labelled_data is True:
            self.human_tweets_df = self.tweets_df.loc[np.where((self.tweets_df['Label'] == 0))[0]]
            self.bot_tweets_df = self.tweets_df.loc[np.where((self.tweets_df['Label'] == 1))[0]]
        else:
            if bot_classifier == 'botaegis':
                self.human_tweets_df = self.tweets_df.loc[np.where((self.tweets_df['Bot scores'] < threshold))[0]]
                self.bot_tweets_df = self.tweets_df.loc[np.where((self.tweets_df['Bot scores'] >= threshold))[0]]
            elif bot_classifier == 'botometer_astroturf':
                self.human_tweets_df = self.tweets_df.loc[np.where((self.tweets_df['Astroturf'] < threshold))[0]]
                self.bot_tweets_df = self.tweets_df.loc[np.where((self.tweets_df['Astroturf'] >= threshold))[0]]
            elif bot_classifier == 'botometer_overall':
                self.human_tweets_df = self.tweets_df.loc[np.where((self.tweets_df['Overall'] < threshold))[0]]
                self.bot_tweets_df = self.tweets_df.loc[np.where((self.tweets_df['Overall'] >= threshold))[0]]


        self.cv = CountVectorizer(max_df=0.75, min_df=2)
        self.human_cv = CountVectorizer(max_df=0.75, min_df=2)
        self.bot_cv = CountVectorizer(max_df=0.75, min_df=2)

        self.dtm = self.cv.fit_transform(self.tweets_df['Tokens'])
        self.human_dtm = self.human_cv.fit_transform(self.human_tweets_df['Tokens'])
        self.bot_dtm = self.bot_cv.fit_transform(self.bot_tweets_df['Tokens'])

        #print(self.dtm)

        # We can also get all those words using the get_feature_names() function
        feature_names = self.cv.get_feature_names()
        human_feature_names = self.human_cv.get_feature_names()
        bot_feature_names = self.bot_cv.get_feature_names()
        print(len(feature_names))
        print(len(human_feature_names)) # show the total number of distinct words
        print(len(bot_feature_names))

        print(feature_names)

    def run_LDA(self, num_LDA_topics, threshold, labelled_data, LDA_tweets_out_file_name_desc, bot_classifier = 'botaegis'): #TODO add threshold as arg
        import json
        print("Running LDA analysis...")
        
        self.LDA_model = LatentDirichletAllocation(n_components=num_LDA_topics, max_iter=30, random_state=42, verbose = 2)
        self.LDA_model.fit(self.dtm)

        self.human_LDA_model = LatentDirichletAllocation(n_components=num_LDA_topics, max_iter=30, random_state=42, verbose = 2)
        self.human_LDA_model.fit(self.human_dtm)

        self.bot_LDA_model = LatentDirichletAllocation(n_components=num_LDA_topics, max_iter=30, random_state=42, verbose = 2)
        self.bot_LDA_model.fit(self.bot_dtm)

        # Pick a single topic 
        a_topic = self.human_LDA_model.components_[0]

        # Get the indices that would sort this array
        a_topic.argsort()

        topic_words_list = []
        for i, topic in enumerate(self.LDA_model.components_):
            print("THE HUMAN TOP {} WORDS FOR TOPIC #{}".format(20, i))
            topic_words = [self.cv.get_feature_names()[index] for index in topic.argsort()[-50:]]
            topic_words_list.append(topic_words)

            print(topic_words)
            print("\n")
        
        human_topic_words_list = []
        for i, topic in enumerate(self.human_LDA_model.components_):
            print("THE HUMAN TOP {} WORDS FOR TOPIC #{}".format(20, i))
            human_topic_words = [self.human_cv.get_feature_names()[index] for index in topic.argsort()[-50:]]
            human_topic_words_list.append(human_topic_words)
            print(human_topic_words)
            print("\n")

        bot_topic_words_list = []
        num_bot_topics = len(self.bot_LDA_model.components_)
        print("LLL", num_bot_topics)
        print(self.bot_LDA_model.components_)
        for i, topic in enumerate(self.bot_LDA_model.components_):
            print("THE BOT TOP {} WORDS FOR TOPIC #{}".format(20, i))
            bot_topic_words = [self.bot_cv.get_feature_names()[index] for index in topic.argsort()[-50:]]
            bot_topic_words_list.append(bot_topic_words)
            print(bot_topic_words)
            print("\n")
        
        with open("total_topic_words.json", 'w') as f:
            json.dump(topic_words_list, f, indent=2)

        with open("human_topic_words.json", 'w') as f:
            json.dump(human_topic_words_list, f, indent=2)

        with open("bot_topic_words.json", 'w') as f:
            json.dump(bot_topic_words_list, f, indent=2)

        final_topics = self.LDA_model.transform(self.dtm)
        human_final_topics = self.human_LDA_model.transform(self.human_dtm)
        bot_final_topics = self.bot_LDA_model.transform(self.bot_dtm)
        print(human_final_topics.shape)

        print(final_topics)
        print(human_final_topics)
        print(bot_final_topics)

        print(self.tweets_df)
        print(type(final_topics.argmax(axis=1)))
        print(final_topics.argmax(axis=1).shape)

        self.tweets_df["Overall topic number"] = -99
        self.tweets_df["Human topic number"] = -99
        self.tweets_df["Bot topic number"] = -99

        # We can use labelled truth bot score data or bot scores we have found 
        self.tweets_df["Overall topic number"] = final_topics.argmax(axis=1)
        if labelled_data is True:
            self.tweets_df["Human topic number"].loc[np.where((self.tweets_df['Label'] == 0))[0]] = human_final_topics.argmax(axis=1)
            self.tweets_df["Bot topic number"].loc[np.where((self.tweets_df['Label'] == 1))[0]] = bot_final_topics.argmax(axis=1)
        else:
            if bot_classifier == 'botaegis':
                self.tweets_df["Human topic number"].loc[np.where((self.tweets_df['Bot scores'] < threshold))[0]] = human_final_topics.argmax(axis=1)
                self.tweets_df["Bot topic number"].loc[np.where((self.tweets_df['Bot scores'] >= threshold))[0]] = bot_final_topics.argmax(axis=1)
            elif bot_classifier == 'botometer_astroturf':
                self.tweets_df["Human topic number"].loc[np.where((self.tweets_df['Astroturf'] < threshold))[0]] = human_final_topics.argmax(axis=1)
                self.tweets_df["Bot topic number"].loc[np.where((self.tweets_df['Astroturf'] >= threshold))[0]] = bot_final_topics.argmax(axis=1)
            elif bot_classifier == 'botometer_overall':
                self.tweets_df["Human topic number"].loc[np.where((self.tweets_df['Overall'] < threshold))[0]] = human_final_topics.argmax(axis=1)
                self.tweets_df["Bot topic number"].loc[np.where((self.tweets_df['Overall'] >= threshold))[0]] = bot_final_topics.argmax(axis=1)


        # Setup directories for saving outputs:
        #TODO assumes data saved in data/analysed_tweets/<keywords>/
        data_out_dir = self.LDA_tweets_in_file_path.split("/")[0] + '/' + self.LDA_tweets_in_file_path.split("/")[1]
        keywords_dir = self.LDA_tweets_in_file_path.split("/")[2]
        
        print(data_out_dir + '/' + keywords_dir)
        if not os.path.exists(data_out_dir + '/' + keywords_dir):
            os.makedirs(data_out_dir + '/' + keywords_dir)

        self.tweets_df.to_csv(data_out_dir + '/' + keywords_dir + "/" + LDA_tweets_out_file_name_desc + '.csv')
        print("Saved tweets with classifier bot scores to", str(data_out_dir + '/' + keywords_dir + "/" + LDA_tweets_out_file_name_desc + '.csv'))

    def visualise(self):
        import pyLDAvis.sklearn

        # Create the panel for the visualization
        panel = pyLDAvis.sklearn.prepare(self.LDA_model, self.dtm, self.cv, mds='tsne')
        pyLDAvis.save_html(panel, 'total_pyladvis_.html')

        human_panel = pyLDAvis.sklearn.prepare(self.human_LDA_model, self.human_dtm, self.human_cv, mds='tsne')
        pyLDAvis.save_html(human_panel, 'human_pyladvis_.html')

        bot_panel = pyLDAvis.sklearn.prepare(self.bot_LDA_model, self.bot_dtm, self.bot_cv, mds='tsne')
        pyLDAvis.save_html(bot_panel, 'bot_pyladvis_.html')