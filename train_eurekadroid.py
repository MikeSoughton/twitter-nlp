#!/usr/bin/env python
"""
Code to call the TrainEurekaDroid class and train the human/bot classifier model. 
"""

import argparse
import json
import eurekadroid


def train_bot_classifier():
    # Load main config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file)

    # Data configurations
    human_tweets_dir = config["DataCfg"]["human_tweets_dir"]
    human_tweets_files = config["DataCfg"]["human_tweets_files"]
    bot_tweets_dir = config["DataCfg"]["bot_tweets_dir"]
    bot_tweets_files = config["DataCfg"]["bot_tweets_files"]
    data_used_name = config["DataCfg"]["data_used_name"]
    filter_on_english = config["DataCfg"]["filter_on_english"]
    test_size = config["DataCfg"]["test_size"]
    reduced_data_size = config["DataCfg"]["reduced_data_size"]
    
    # Vectorisation parameters
    max_features = config["VectorisationParams"]["max_features"]
    max_len = config["VectorisationParams"]["max_len"]

    # Model parameters
    model_dir = config["ModelCfg"]["model_dir"]
    model_name = config["ModelCfg"]["model_name"]
    load_weights = config["ModelCfg"]["load_weights"]
    pretrained_weights_path = config["ModelCfg"]["pretrained_weights_path"]
    epochs = config["ModelCfg"]["epochs"]
    batch_size = config["ModelCfg"]["batch_size"]


    # Initialise the model training class
    TrainEurekaDroid = eurekadroid.TrainEurekaDroid()

    # Data preparations
    TrainEurekaDroid.read_data(filter_on_english, human_tweets_dir, bot_tweets_dir, human_tweets_files = human_tweets_files, bot_tweets_files = bot_tweets_files)
    TrainEurekaDroid.prepare_data(max_features, max_len, test_size, reduced_data_size = reduced_data_size)

    # Build the model based on the models found in 'models/'
    TrainEurekaDroid.build_model(model_name, max_features = max_features, max_len = max_len)

    # Train the model
    TrainEurekaDroid.train_model(model_dir, model_name, data_used_name, epochs, batch_size, load_weights, pretrained_weights_path = pretrained_weights_path)
    
    # Optional - test the model #TODO add optional argument
    TrainEurekaDroid.test_model(batch_size = batch_size)

if __name__ == '__main__':

    # Just speciffying config file as default arg here
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default = 'configs/train_bot_classifier_config.json', help = 'str: train bot classifier configuration file path')
    args = parser.parse_args()

    train_bot_classifier()