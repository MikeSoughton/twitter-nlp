#!/usr/bin/env python
"""
D
"""

import argparse
import json
import zucc
#import models 


def run_bot_classifier():
    # Load main config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file)

    # Data configurations
    human_tweets_dir = config["DataCfg"]["human_tweets_dir"]
    human_tweets_files = config["DataCfg"]["human_tweets_files"]
    bot_tweets_dir = config["DataCfg"]["bot_tweets_dir"]
    bot_tweets_files = config["DataCfg"]["bot_tweets_files"]
    reduced_data_size = config["DataCfg"]["reduced_data_size"]
    
    # Vectorisation parameters
    max_features = config["VectorisationParams"]["max_features"]
    max_len = config["VectorisationParams"]["max_len"]


    print(human_tweets_dir, human_tweets_files, bot_tweets_dir, bot_tweets_files)
    print(max_features, max_len)

    ZuccDestroyer = zucc.ZuccDestroyer(max_features, max_len)

    #ZuccDestroyer.read_data(human_tweets_dir, bot_tweets_dir, human_tweets_files = human_tweets_files, bot_tweets_files = bot_tweets_files, reduced_data_size = reduced_data_size)
    #ZuccDestroyer.prepare_data(max_features, max_len)

    model_name = "solo_cnn"
    ZuccDestroyer.build_model2(model_name, max_features = max_features, max_len = max_len)
    


if __name__ == '__main__':

    # Just speciffying config file as default arg here
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default = 'train_bot_classifier_config.json', help = 'str: train bot classifier configuration file path')
    args = parser.parse_args()

    run_bot_classifier()