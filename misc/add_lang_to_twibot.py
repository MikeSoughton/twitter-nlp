#!/usr/bin/env python
"""
The Twibot dataset has a item for language of the user account but it is set to None for all accounts.
We can do one better and get the language for each tweet using Spacy.

Also in comments is how to remove urls from tweets.
"""

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import json

exclude_rt = True
twibot_train_file_path = '../data/Twibot-20/train.json'
twibot_test_file_path = '../data/Twibot-20/test.json'
twibot_val_file_path = '../data/Twibot-20/dev.json'

#with open(twibot_train_file_path) as f:
#    train_file = json.load(f)
with open(twibot_test_file_path) as f:
    test_file = json.load(f)
#with open(twibot_val_file_path) as f:
#    val_file = json.load(f)

# Setup Spacy
def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

def add_language(file):
    for i, user in enumerate(file):
        user_id = user["profile"]["id_str"]
        username = user["profile"]["screen_name"]
        display_name = user["profile"]["name"]

        try:
            tweets = user["tweet"]
            if exclude_rt is True:
                try:
                    tweets = [tweet.split('RT @')[1].split(':')[1] for tweet in tweets]
                    
                    # Remove url links
                    #tweets = [re.sub(r'http\S+', '', tweet) for tweet in tweets]
                except IndexError: # Except occurs if tweet is not a retweet (retweets start 'RT @')
                    pass
            
            # We'll make a list 
            lang_list = []
            for j, tweet in enumerate(tweets):
                #print(tweet + "\n")
                doc = nlp(tweet)
                #print(doc._.language)
                lang = doc._.language['language']
                lang_list.append(lang)

                if lang != "en":
                    print(tweet)
                    print(i, j, lang + "\n")
            
            user["tweet_lang"] = lang_list

                

        except TypeError: # Except occurs if user has no tweets
            pass

    return file

train_file_out = add_language(test_file)
test_file_out = add_language(test_file)
val_file_out = add_language(test_file)

twibot_train_out_file_path = '../data/Twibot-20/train_lang.json'
twibot_test_out_file_path = '../data/Twibot-20/test_lang.json'
twibot_val_out_file_path = '../data/Twibot-20/val_lang.json'

with open(twibot_test_out_file_path, 'w') as output_json:
    json.dump(test_file_out, output_json, indent=4)

#with open(twibot_train_out_file_path, 'w') as output_json:
#    json.dump(train_file_out, output_json, indent=4)

#with open(twibot_val_out_file_path, 'w') as output_json:
#    json.dump(val_file_out, output_json, indent=4)
    
print("Saved output file as '{}'".format(twibot_test_out_file_path))