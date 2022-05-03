#!/usr/bin/env python
"""
Code to run botometer over tweets and obtain the bot scores it predicts. 
"""

import argparse
import json
import eurekadroid


def run_botometer():
    # Load main config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file)

    # Data configurations
    restore_checkpoints = config["DataCfg"]["restore_checkpoints"]
    raw_tweets_dir = config["DataCfg"]["raw_tweets_dir"]
    tweets_file_name_desc = config["DataCfg"]["tweets_file_name_desc"]
    checkpoints_tweets_dir = config["DataCfg"]["checkpoints_tweets_dir"]
    data_out_dir = config["DataCfg"]["data_out_dir"]

    # API keys configurations
    twitter_api_keys_config = config["APIkeysCfg"]["twitter_api_keys_config"]
    rapid_api_keys_config = config["APIkeysCfg"]["rapid_api_keys_config"]
    
    # Manual Botometer limits (a saftey net to ensure you don't end up overspending - but always be careful!)
    botometer_max_limit = config["BotometerCfg"]["botometer_max_limit"]
    bot_check_count = config["BotometerCfg"]["bot_check_count"] #TODO this is just entered manually - it could be automated
    checkpoints = config["BotometerCfg"]["checkpoints"]
    start_point = config["BotometerCfg"]["start_point"]
    end_point = config["BotometerCfg"]["end_point"]

    # Initialise
    RunBotometer = eurekadroid.RunBotometer(botometer_max_limit, bot_check_count)

    # Read data
    if restore_checkpoints is True:
        # This is configured for the endpoint - it could be made to work for the checkpoints as well,
        # but I don't want the config file to be too large and they are backups only anyway
        tweets_file_path = checkpoints_tweets_dir + tweets_file_name_desc + "_botometer_" + str(start_point) + "endpoint.csv"
        print("aaaaa", tweets_file_path)
    else:
        tweets_file_path = raw_tweets_dir + tweets_file_name_desc + ".csv"
    
    RunBotometer.read_data(tweets_file_path)

    # Get accounts list
    RunBotometer.get_accounts()

    # Setup botometer
    RunBotometer.setup_botometer(twitter_api_keys_config, rapid_api_keys_config)

    # Run botometer
    RunBotometer.run_botometer(data_out_dir, tweets_file_name_desc, checkpoints, start_point, end_point)


    



if __name__ == '__main__':

    # Just speciffying config file as default arg here
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default = 'configs/run_botometer_config.json', help = 'str: train bot classifier configuration file path')
    args = parser.parse_args()

    run_botometer()