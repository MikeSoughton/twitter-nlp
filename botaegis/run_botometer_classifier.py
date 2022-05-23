import enum
import os
import json
import botometer

import numpy as np 
import pandas as pd


class RunBotometer():
    def __init__(self, botometer_max_limit, botometer_check_count):
        self.botometer_max_limit = botometer_max_limit
        self.botometer_check_count = botometer_check_count

    def read_data(self, tweets_file_path):
        print("Reading in data...")

        self.tweets_file_path = tweets_file_path
        self.tweets_df = pd.read_csv(tweets_file_path)

        # Drop any NaN values if they exist
        self.tweets_df = self.tweets_df.dropna(subset=['Username'], how='all')
        #self.tweets_df = self.tweets_df.dropna(subset=['Text'], how='all')
        #self.tweets_df = self.tweets_df.reset_index(drop=True) # Can be used if desired


    def get_accounts(self, fill_duplicates = False):
        print("Getting accounts...")
        """
        Simply gets the accounts into a list which can be fed into botometer.
        If fill_duplicates = True, give each duplicate account in the dataset
        an empty value of ''. This is only really useful for when running over
        the training dataset.
        """
        self.accounts = self.tweets_df["Username"].copy().to_numpy()

        # Botometer requires the @ symbol
        for idx,account in enumerate(self.accounts):
            self.accounts[idx] = "@" + account

        # Fill duplicate accounts with empty string - these will be ignored by the if statement in run_botometer
        if fill_duplicates is True:
            seen = set()
            for idx, account in enumerate(self.accounts):
                if account in seen:
                    self.accounts[idx] = ''
                else:
                    seen.add(account)


    def setup_botometer(self, twitter_api_keys_config, rapid_api_keys_config):
        print("Setting up botometer...")

        # Load my API keys
        with open(twitter_api_keys_config) as twitter_cfg_file:
            twitter_config = json.load(twitter_cfg_file)

        with open(rapid_api_keys_config) as rapid_api_cfg_file:
            rapidapi_config = json.load(rapid_api_cfg_file)

        consumer_key = twitter_config["API_key"]
        consumer_secret = twitter_config["API_secret"]
        access_token = twitter_config["access_token"]
        access_token_secret = twitter_config["access_secret"]

        rapidapi_key = rapidapi_config["API_key"]


        twitter_app_auth = {
                            'consumer_key': consumer_key,
                            'consumer_secret': consumer_secret,
                            'access_token': access_token,
                            'access_token_secret': access_token_secret
                        }
        botometer_api_url = "https://botometer-pro.p.rapidapi.com"

        self.bom = botometer.Botometer(
                        wait_on_ratelimit = True,
                        botometer_api_url=botometer_api_url,
                        rapidapi_key = rapidapi_key,
                        **twitter_app_auth)


    def run_botometer(self, data_out_dir, tweets_file_name_desc, checkpoints, start_point, end_point):
        print("Running botometer...")

        # Setup directories for saving outputs:
        #TODO assumes data saved in data/scraped_tweets/<keywords>/
        keywords_dir = self.tweets_file_path.split("/")[2]
        #data_file_name = self.tweets_file_path.split("/")[len(self.tweets_file_path.split("/")) - 1]
        data_file_name = tweets_file_name_desc


        if not os.path.exists(data_out_dir + keywords_dir):
            os.makedirs(data_out_dir + keywords_dir)

        if not os.path.exists(data_out_dir + keywords_dir + "/botometer_checkpoints"):
            os.makedirs(data_out_dir + keywords_dir + "/botometer_checkpoints")

        # Run botometer, looping over all accounts in the accounts list

        #TODO This line is an alternative way of looping, but offers less flexibility. Delete when the current method is tested
        #for idx, (screen_name, result) in enumerate(self.bom.check_accounts_in(self.accounts[start_point:])):
        for idx, account in enumerate(self.accounts[start_point:]):
            #print(result)
            print(idx, idx + start_point, account)
            import tweepy
            if account != '':
                try:
                    result = self.bom.check_account(account)
                except tweepy.error.TweepError as e:
                    if "User not found" in str(e):
                        print("User account not found, passing")
                    else:
                        print("Unexpected error")

                try:
                    #print(result)
                    self.tweets_df.loc[idx + start_point, "Astroturf"] = result['raw_scores']['english']['astroturf']
                    self.tweets_df.loc[idx + start_point, "Fake follower"] = result['raw_scores']['english']['fake_follower']
                    self.tweets_df.loc[idx + start_point, "Financial"] = result['raw_scores']['english']['financial']
                    self.tweets_df.loc[idx + start_point, "Other"] = result['raw_scores']['english']['other']
                    self.tweets_df.loc[idx + start_point, "Overall"] = result['raw_scores']['english']['overall']
                    self.tweets_df.loc[idx + start_point, "Self declared"] = result['raw_scores']['english']['self_declared']
                    self.tweets_df.loc[idx + start_point, "Spammer"] = result['raw_scores']['english']['spammer']
                    self.tweets_df.loc[idx + start_point, "Cap"] = result['cap']['english']
                except KeyError:
                    if "User not found" in result["error"]:
                        print("User account not found, passing2")
                    else:
                        print("Unexpected error2")

                # Our saftey net will break if we reach the max daily limit
                #TODO you could put another statement here to wait until the next day without breaking
                # and continue then, but I'd rather not and run it manually
                if self.botometer_check_count == (self.botometer_max_limit - 1):
                    print("Reached max daily limit on bot checks, stopping")
                    break
                self.botometer_check_count += 1

            if (idx > 1) & ((idx - 1) % checkpoints == 0):
                self.tweets_df.to_csv(data_out_dir + keywords_dir + "/botometer_checkpoints/" + data_file_name + '_botometer_' + str(idx + start_point) + 'checkpoint.csv')
                print("Saved tweets checkpoint with botometer scores with " + str(idx + start_point) + " accounts checked")

            if (idx + start_point) == end_point:
                print("Reached end point of" + str(end_point), ", breaking.")
                break
            
            #TODO this statement is not yet tested
            if (idx + start_point) == self.tweets_df.shape[0]:
                print("Finished running over file, ending")
                break
            

        print(self.tweets_df)

        self.tweets_df.to_csv(data_out_dir + keywords_dir + "/botometer_checkpoints/" + data_file_name + '_botometer_' + str(end_point) + 'endpoint.csv')
        print("Saved tweets with classifier bot scores to", str(data_out_dir + keywords_dir + "/botometer_checkpoints/" + data_file_name + '_botometer_' + str(end_point) + 'endpoint.csv'))
        
            

