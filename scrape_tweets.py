#!/usr/bin/env python
"""
Run this to scrape tweets. Tweets can be scraped either by:
    mode 1: #TODO tweets containing specific keywords within a certain date range
    mode 2: tweets from accounts in a list of given user ids
"""

import argparse
import json
import tweetsearcher

def get_tweets_by_keyword():
    pass

def get_tweets_by_id():

    # Data configurations
    user_ids_dir = config["Mode2Cfg"]["user_ids_dir"]
    user_ids_file = config["Mode2Cfg"]["user_ids_file"]
    data_out_dir = config["Mode2Cfg"]["data_out_dir"]
    data_out_file_desc = config["Mode2Cfg"]["data_out_file_desc"]
    max_tweets_per_user = config["Mode2Cfg"]["max_tweets_per_user"]

    # Initialise the model training class
    GetTweetsById = tweetsearcher.GetTweetsById(max_tweets_per_user)

    # Load in user ids of human accounts
    GetTweetsById.load_ids(user_ids_dir, user_ids_file)

    # Download and save
    GetTweetsById.download(data_out_dir, data_out_file_desc)

    

if __name__ == '__main__':

    # Just speciffying config file as default arg here
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default = 'scrape_tweets_config.json', help = 'str: train bot classifier configuration file path')
    args = parser.parse_args()

    # Load main config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file)
    
    mode = config["Mode"]

    if mode == 2:
        get_tweets_by_id()
