#!/usr/bin/env python
"""
Code to call the RunBotAegis class and run the human/bot classifier model. 
"""

import argparse
import json
import botaegis


def run_bot_classifier():
    # Load main config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file)

    # Data configurations
    tweets_file_path = config["DataCfg"]["tweets_file_path"]
    tokenizer_file_path = config["DataCfg"]["tokenizer_file_path"]
    train_data_used_name = config["DataCfg"]["train_data_used_name"]
    data_out_dir = config["DataCfg"]["data_out_dir"]

    
    # Vectorisation parameters
    max_features = config["VectorisationParams"]["max_features"]
    max_len = config["VectorisationParams"]["max_len"]

    # Model parameters
    model_dir = config["ModelCfg"]["model_dir"]
    model_name = config["ModelCfg"]["model_name"]
    trained_weights_path = config["ModelCfg"]["trained_weights_path"]
    batch_size = config["ModelCfg"]["batch_size"]


    # Initialise the model training class
    RunBotAegis = botaegis.RunBotAegis()

    RunBotAegis.read_data(tweets_file_path)

    RunBotAegis.prepare_data(tokenizer_file_path, max_len)

    RunBotAegis.build_model(model_name, max_features = max_features, max_len = max_len)

    RunBotAegis.run_model(model_dir, model_name, train_data_used_name, trained_weights_path, data_out_dir, batch_size = batch_size)



if __name__ == '__main__':

    # Just speciffying config file as default arg here
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default = 'configs/run_bot_classifier_config.json', help = 'str: train bot classifier configuration file path')
    args = parser.parse_args()

    run_bot_classifier()