#!/usr/bin/env python
"""
Code to run botometer over tweets and obtain the bot scores it predicts. 
"""

import argparse
import json
import botaegis


def run_get_sentiment():
    # Load main config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file)

    # Data configurations
    raw_tweets_dir = config["DataCfg"]["raw_tweets_dir"]
    tweets_file_name_desc = config["DataCfg"]["tweets_file_name_desc"]

    tweets_file_path = raw_tweets_dir + tweets_file_name_desc + ".csv"


    # Initialise
    GetSentiment = botaegis.GetSentiment()

    GetSentiment.read_data(tweets_file_path)

    GetSentiment.prepare_data()

    GetSentiment.run_sentiment()

    



if __name__ == '__main__':

    # Just speciffying config file as default arg here
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default = 'configs/get_sentiment_config.json', help = 'str: train bot classifier configuration file path')
    args = parser.parse_args()

    run_get_sentiment()