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
    tweets_dir = config["DataCfg"]["tweets_dir"]
    tweets_in_file_name_desc = config["DataCfg"]["tweets_in_file_name_desc"]
    data_out_dir = config["DataCfg"]["data_out_dir"]
    sentiment_tweets_out_file_name_desc = config["DataCfg"]["sentiment_tweets_out_file_name_desc"]
    LDA_tweets_in_file_path = config["DataCfg"]["LDA_tweets_in_file_path"]
    LDA_tweets_out_file_name_desc = config["DataCfg"]["LDA_tweets_out_file_name_desc"]

    # Sentiment configurations
    get_sentiment = config["SentimentCfg"]["get_sentiment"]
    run_LDA = config["SentimentCfg"]["run_LDA"]
    bot_threshold = config["SentimentCfg"]["bot_threshold"]
    num_LDA_topics = config["SentimentCfg"]["num_LDA_topics"]

    # Get sentiment scores
    if get_sentiment is True:
        tweets_file_path = tweets_dir + tweets_in_file_name_desc + ".csv"
        GetSentiment = botaegis.GetSentiment(tweets_file_path)

        GetSentiment.read_data()

        GetSentiment.prepare_data()

        GetSentiment.run_sentiment(data_out_dir, sentiment_tweets_out_file_name_desc)

    # LDA analysis
    if run_LDA is True:
        LDA_Decomposition = botaegis.LDA_Decomposition(LDA_tweets_in_file_path)

        LDA_Decomposition.read_data()

        LDA_Decomposition.filter_stopwords()

        LDA_Decomposition.setup_countvectorizer(bot_threshold)

        LDA_Decomposition.run_LDA(num_LDA_topics, LDA_tweets_out_file_name_desc)

        LDA_Decomposition.visualise()

    



if __name__ == '__main__':

    # Just speciffying config file as default arg here
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default = 'configs/run_sentiment_config.json', help = 'str: train bot classifier configuration file path')
    args = parser.parse_args()

    print("Running with config", args.config)

    run_get_sentiment()