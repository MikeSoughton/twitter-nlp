#!/usr/bin/env python
"""
Simple code for investigating our dataset. This file is very messy and is not maintained - it is purely for
quick and dirty work. Various tasks are put here together. Don't bother looking at this if you don't know 
what you want to find in here, but I do keep this file here as the code may be useful elsewhere.
"""

import argparse
import json
import pandas as pd
import numpy as np


def main(tweets_file_path):

    def check_scores():
        """
        Checks scores of 
        """
        # Load dataset
        tweets_df = pd.read_csv(tweets_file_path)

        # Drop NaN value
        tweets_df = tweets_df.dropna(subset=['Overall'], how='all')        
        tweets_df.reset_index(drop=True, inplace=True)

        # Save trimmed version 
        #tweets_df.to_csv('a.csv')
        print(tweets_df['Text'].loc[np.where((tweets_df['Overall'] > 0.75))[0]])
        print(len(tweets_df['Text'].loc[np.where((tweets_df['Overall'] > 0.75))[0]]))
        print(len(tweets_df['Text']))

        print(len(tweets_df['Text'].loc[np.where((tweets_df['Astroturf'] > 0.75))[0]]))
        print(len(tweets_df['Text']))

        print(len(tweets_df['Text'].loc[np.where((tweets_df['Bot scores'] > 0.75))[0]]))
        print(len(tweets_df['Text']))

        print(tweets_df)
    
    def get_reddit_data():
        """
        Gets and filters the data from 
        https://www.reddit.com/r/datasets/comments/8s6nqz/all_verified_twitter_users_100_complete_in_ndjson/
        """
        # Extra stuff
        import os
        print(os.getcwd())
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        df = pd.read_json(dir_path + '/TU_verified.ndjson', lines=True)
        df2 = df[['name','screen_name','id','followers_count']].copy()
        print(df2)
        del df

        # Filter by human names - since there are SO many organisations in this dataset
        # we could not possibly hope to filter them all. Instead do it the other way around
        names_path = 'human_names_dict.txt'
        with open(names_path, encoding="utf8", errors='ignore') as f:
            lines = f.readlines()

        # Names file - I found this one, which includes names from many cultures
        # But it may have been better just to use English ones to avoid cases like 'You' being included
        # and so not filtering out things like 'youtube'
        names_list = []
        import string
        for idx, line in enumerate(lines[362:]): # This file is so weirdly laid out, really!
            #print(line[3:].split(' ')[0])
            name = line[3:].split(' ')[0]
            char_set = string.ascii_letters
            if all((True if x in char_set else False for x in name)) is True:
                names_list.append(name)
        
        import gc

        df_dropped = df2.copy()
        #for name in names_list:
        #    print(name)
        #    df_dropped = df_dropped.drop(df_dropped[df_dropped['name'].str.lower().str.contains(str.lower(name))].index)
        #    gc.collect()

        # It takes too much memory to loop through the dataset and drop line-by-line. This line is much faster
        name_mask = df2['name'].str.lower().str.split(' ').str[0].isin([x.lower() for x in names_list])
        df2 = df2[name_mask]

        print(df2)

        # Let's also filter out organisations form our list to double-check:
        organisations_file_path = "data/organisations.csv"
        organisations_list = list(np.genfromtxt(organisations_file_path, delimiter=',', dtype=str, encoding='utf-8'))

        for organisation_name in organisations_list:
            df2 = df2.drop(df2[df2['name'].str.lower().str.contains(str.lower(organisation_name))].index)

        print(df2)
        df2 = df2.reset_index(drop=True)
        df2.to_csv('data/verified-reddit/verified_reddit_latin.csv')



    get_reddit_data()



if __name__ == '__main__':

    # Just speciffying config file as default arg here
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default = 'data/checked_tweets/verified_2019_200peruser/verified_2019_200peruser_botometer.csv')
    args = parser.parse_args()

    tweets_file_path = args.data
    tweets_file_path = "data/checked_tweets/verified_2019_200peruser/verified_2019_200peruser_botometer_classi.csv"

    main(tweets_file_path)