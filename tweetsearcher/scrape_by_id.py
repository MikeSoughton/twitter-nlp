"""
Download tweets given user ids.
"""

import os
import snscrape.modules.twitter as sntwitter
import pandas as pd
from tqdm import tqdm

class GetTweetsById():
    def __init__(self, max_tweets_per_user):
        self.max_tweets_per_user = max_tweets_per_user

    def load_ids(self, users_id_dir, users_id_file):
        file_path = users_id_dir + users_id_file
        file_extension = os.path.splitext(users_id_file)[1]

        # Can add other options in elif if necessary
        if file_extension == '.tsv':
            user_ids_df = pd.read_csv(file_path, delimiter='\t', header=None)
            self.user_ids = user_ids_df.iloc[:, 0]. tolist() # A little bit of hardcoding on the column here

        elif file_extension == '.csv':
            user_ids_df = pd.read_csv(file_path, delimiter=',', header=None)
            self.user_ids = user_ids_df.iloc[143144:144144, 3]. tolist() # A little bit of hardcoding on the rows and column here
            print(self.user_ids)

    def download(self, lang, data_out_dir, data_out_file_desc):
        """
        Downloads tweets and saves the output.
        """
        tweets_list = []
        for i, user_id in enumerate(self.user_ids):
            pbar = tqdm(sntwitter.TwitterUserScraper(str(user_id), isUserId = True).get_items())

            # Because snscrape randomly started being super slow
            from interruptingcow import timeout
            try:
                with timeout(15, exception=RuntimeError):
                    try:
                        for j,tweet in enumerate(pbar):
                            
                            pbar.set_postfix_str('User %s, tweet %s' % (i, j))
                            if j >= self.max_tweets_per_user:
                                break
                            
                            if tweet.lang == lang:
                                tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.likeCount, tweet.user.displayname, tweet.lang, tweet.user.created])
                        
                    except KeyError:
                        pass                           
            except RuntimeError:
                print("snscrape stuck, ignoring user")
                continue


            
        # Creating a dataframe from the tweets list above
        human_tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'Like Count', 'Display Name', 'Language', 'User Created'])
        human_tweets_df.reset_index(drop=True, inplace=True)

        human_tweets_df.to_csv(data_out_dir + data_out_file_desc + "_" + str(self.max_tweets_per_user) + "peruser.csv")


 