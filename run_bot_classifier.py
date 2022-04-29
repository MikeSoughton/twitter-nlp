#!/usr/bin/env python
"""
Code to call the 'zucc' (NAME TO BE CHANGED) class and run the human/bot classifier model. 
"""

import argparse
import json
import zucc


def run_bot_classifier():
    # Load main config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file)

    # Data configurations
    tweets_file_path = config["DataCfg"]["tweets_file_path"]
    tokenizer_file_path = config["DataCfg"]["tokenizer_file_path"]
    train_data_used_name = config["DataCfg"]["train_data_used_name"]

    
    # Vectorisation parameters
    max_features = config["VectorisationParams"]["max_features"]
    max_len = config["VectorisationParams"]["max_len"]

    # Model parameters
    model_dir = config["ModelCfg"]["model_dir"]
    model_name = config["ModelCfg"]["model_name"]
    load_weights = config["ModelCfg"]["load_weights"]
    trained_weights_path = config["ModelCfg"]["trained_weights_path"]
    epochs = config["ModelCfg"]["epochs"]
    batch_size = config["ModelCfg"]["batch_size"]


    # Initialise the model training class
    RunZucc = zucc.RunZucc() # #COME UP WITH NEW NAME

    RunZucc.read_data(tweets_file_path)

    RunZucc.prepare_data(tokenizer_file_path, max_len)

    RunZucc.build_model(model_name, max_features = max_features, max_len = max_len)

    RunZucc.run_model(model_dir, model_name, train_data_used_name, trained_weights_path, batch_size = batch_size)



if __name__ == '__main__':

    # Just speciffying config file as default arg here
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default = 'run_bot_classifier_config.json', help = 'str: train bot classifier configuration file path')
    args = parser.parse_args()

    run_bot_classifier()