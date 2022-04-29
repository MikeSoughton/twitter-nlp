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

    # Data configurations
    data_out_dir = config["Mode1Cfg"]["DataCfg"]["data_out_dir"]
    organisations_file_path = config["Mode1Cfg"]["DataCfg"]["organisations_file_path"]
    prescraped_tweets_df_file_path = config["Mode1Cfg"]["DataCfg"]["prescraped_tweets_df_file_path"]

    # Scraping configurations
    scrape = config["Mode1Cfg"]["ScrapingCfg"]["scrape"]
    keywords = config["Mode1Cfg"]["ScrapingCfg"]["keywords"]
    num_tweets_per_day = config["Mode1Cfg"]["ScrapingCfg"]["num_tweets_per_day"]
    start_date = config["Mode1Cfg"]["ScrapingCfg"]["start_date"]
    end_date = config["Mode1Cfg"]["ScrapingCfg"]["end_date"]
    lang = config["Mode1Cfg"]["ScrapingCfg"]["language"]
    user_created_after = config["Mode1Cfg"]["ScrapingCfg"]["user_created_after"]

    # Initialise the tweet scraper class
    GetTweetsByKeyword = tweetsearcher.GetTweetsByKeyword(num_tweets_per_day, start_date, end_date, data_out_dir)

    # Download and save tweets
    if scrape is True:
        GetTweetsByKeyword.scrape_tweets_by_keywords(keywords, lang, user_created_after = user_created_after)

    # Filter out organisations
    GetTweetsByKeyword.filter_organisations(organisations_file_path, prescraped_tweets_df_file_path = prescraped_tweets_df_file_path)


def get_tweets_by_id():

    # Data and scraping configurations
    user_ids_dir = config["Mode2Cfg"]["user_ids_dir"]
    user_ids_file = config["Mode2Cfg"]["user_ids_file"]
    data_out_dir = config["Mode2Cfg"]["data_out_dir"]
    data_out_file_desc = config["Mode2Cfg"]["data_out_file_desc"]
    max_tweets_per_user = config["Mode2Cfg"]["max_tweets_per_user"]

    # Initialise the tweet scraper class
    GetTweetsById = tweetsearcher.GetTweetsById(max_tweets_per_user)

    # Load in user ids of human accounts
    GetTweetsById.load_ids(user_ids_dir, user_ids_file)

    # Download and save tweets
    GetTweetsById.download(data_out_dir, data_out_file_desc)
    

if __name__ == '__main__':

    # Just speciffying config file as default arg here
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default = 'scrape_tweets_config.json', help = 'str: train bot classifier configuration file path')
    parser.add_argument('-m', '--mode', type=int, default = None, help = 'int: mode to use (see config description), default is None and will be taken from config')
    args = parser.parse_args()

    # Load main config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file)
    
    if args.mode is None:
        mode = config["Mode"]
    else:
        mode = args.mode
    print("mode", mode)
    if mode == 1:
        get_tweets_by_keyword()

    elif mode == 2:
        get_tweets_by_id()
