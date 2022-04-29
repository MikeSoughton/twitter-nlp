"""
Download tweets given a list of keyword(s). A list...
"""

import os
import datetime
import numpy as np

import snscrape.modules.twitter as sntwitter
import pandas as pd
from tqdm import tqdm

class GetTweetsByKeyword():
    def __init__(self, num_tweets_per_day, start_date, end_date, data_out_dir):
        self.num_tweets_per_day = num_tweets_per_day
        self.data_out_dir = data_out_dir

        self.start_date = start_date
        self.end_date = end_date
        self.since_list, self.until_list = create_date_lists(start_date, end_date)

    def scrape_tweets_by_keywords(self, keywords, lang, user_created_after = None):
    
        """
        Get tweets using snscrape. If user_created_after is specified, only save tweets if the user was 
        created after that date. Date format is YYYY-MM-DD.
        
        If you use you see 'Total suitable tweets found' is fewer than the total, it means you have 
        found all the tweets for that day.
        """
        
        # Loop through each since and until date in the since and until lists, create a dataframe for each day and stitch them together
        tweets_df = pd.DataFrame(columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'Like Count', 'Display Name', 'Language'])
        for day, (since,until) in enumerate(list(zip(self.since_list, self.until_list))):
            print("Day:", day)
            tweets_list = []
            
            # We want to avoid duplicate tweets if going through multiple keywords in the same day
            daily_tweet_list = []        
            for keyword in keywords:
                
                
                # Split the total number of tweets per day evenly over all keywords
                num_tweets = int(np.floor(self.num_tweets_per_day/len(keywords)))

                print("Keyword:", keyword)
                
                pbar = tqdm(sntwitter.TwitterSearchScraper(keyword + ' since:' + since + ' until:' + until + ' lang:' + lang).get_items())
                
                tweet_idx = 0
                #for i,tweet in enumerate(tqdm(sntwitter.TwitterSearchScraper(keyword + ' since:' + since + ' until:' + until + ' lang:' + lang).get_items())):
                for i,tweet in enumerate(pbar):    
                    
                    # Use a custom counter here since we will get fewer tweets when filtering by user creation date
                    if tweet_idx >= num_tweets:
                        break

                    user_created = tweet.user.created
                    user_created = datetime.datetime.strftime(user_created, "%Y-%m-%d")
                    #TODO add a statement to not add users in the organisation list
                    # Would be better to do it here rather than later, but for now it doesn't matter
                    if user_created_after is not None:
                        if user_created > user_created_after:
                            if tweet.id not in daily_tweet_list:
                                tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.likeCount, tweet.user.displayname, tweet.lang, user_created])
                                daily_tweet_list.append(tweet.id)
                                tweet_idx += 1
                                    
                    else:
                        if tweet.id not in daily_tweet_list:
                            tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.likeCount, tweet.user.displayname, tweet.lang, user_created])
                            daily_tweet_list.append(tweet.id)
                            tweet_idx += 1
                        
                    pbar.set_postfix_str('Total suitable tweets found %s/%s' % (tweet_idx, num_tweets))
                        

            # Creating a dataframe from the tweets list above
            tweets_day_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'Like Count', 'Display Name', 'Language', 'User Created'])

            # Stitch the daily dataframes together
            tweets_df = pd.concat([tweets_df, tweets_day_df])
            tweets_df.reset_index(drop=True, inplace=True)

        # Get the names of the keywords for saved file path name
        keywords_string = '_'.join(keywords)
        keywords_string = keywords_string.replace(' ', '_')

        full_data_out_dir = self.data_out_dir + "/" + keywords_string

        if not os.path.exists(full_data_out_dir):
            os.makedirs(full_data_out_dir)

        self.tweets_df_file_path = full_data_out_dir + '/tweets_raw_' + lang + '_df_' + str(self.num_tweets_per_day) + 'dailytweets_' + self.start_date + '_to_' + self.end_date + '.csv'
        tweets_df.to_csv(self.tweets_df_file_path)

        self.tweets_df = tweets_df


    def filter_organisations(self, organisations_file_path, prescraped_tweets_df_file_path = None):
        """
        Filters out organisational accounts by dropping any tweets whose 
        display name or username contains words in the organisations file.
        """

        # Here we give the option to load in a preexisting tweets file
        if prescraped_tweets_df_file_path is not None:
            tweets_df = pd.read_csv(prescraped_tweets_df_file_path, lineterminator='\n')
        else:
            tweets_df = self.tweets_df
        print(tweets_df)

        # First drop any NaN values
        tweets_df = tweets_df.dropna(subset=['Display Name'], how='all')
        tweets_df = tweets_df.dropna(subset=['Username'], how='all')
        print(tweets_df)

        organisations_list = list(np.genfromtxt(organisations_file_path, delimiter=',', dtype=str, encoding='utf-8')[:,0])

        tweets_df_dropped = tweets_df.copy()
        for organisation_name in organisations_list:
            tweets_df_dropped = tweets_df_dropped.drop(tweets_df_dropped[tweets_df_dropped['Display Name'].str.lower().str.contains(str.lower(organisation_name))].index)
            tweets_df_dropped = tweets_df_dropped.drop(tweets_df_dropped[tweets_df_dropped['Username'].str.lower().str.contains(str.lower(organisation_name))].index)
            
        tweets_df = tweets_df_dropped.copy()
        tweets_df = tweets_df.reset_index(drop=True)

        print(tweets_df)

        #TODO Change lang to self.lang so we can replace with 'tweets_' + self.lang + '_df_'
        # to avoid edge case of a keyword ending in df
        if prescraped_tweets_df_file_path is not None:
            tweets_filtered_df_file_path = prescraped_tweets_df_file_path.replace('df_', 'df_individual_')
        else:
            tweets_filtered_df_file_path = self.tweets_df_file_path.replace('df_', 'df_individual_')

        print(tweets_filtered_df_file_path)

        tweets_df.to_csv(tweets_filtered_df_file_path)



        

# A function to create a pair of lists for days to search
def create_date_lists(since_initial, until_final):
    """
    Creates a a pair of lists for since and until dates of the form
    since_list = ['2022-02-24', '2022-02-25', '2022-02-26']
    until_list = ['2022-02-25', '2022-02-26', '2022-02-27']
    """

    diff = datetime.datetime.strptime(until_final, "%Y-%m-%d") - datetime.datetime.strptime(since_initial, "%Y-%m-%d")
    diff = diff.days

    since_initial_datetime = datetime.datetime.strptime(since_initial, "%Y-%m-%d")
    until_initial_datetime = since_initial_datetime + datetime.timedelta(days=1)
    until_initial_datetime = datetime.datetime.strftime(until_initial_datetime, "%Y-%m-%d")

    diff = datetime.datetime.strptime(until_final, "%Y-%m-%d") - datetime.datetime.strptime(since_initial, "%Y-%m-%d")
    diff = diff.days

    since_list = []
    until_list = []
    for day in range(diff):
        since_plus_day = since_initial_datetime + datetime.timedelta(days=day)
        since_plus_day = datetime.datetime.strftime(since_plus_day, "%Y-%m-%d")
        since_list.append(since_plus_day)

        until_plus_day = since_initial_datetime + datetime.timedelta(days=day+1)
        until_plus_day = datetime.datetime.strftime(until_plus_day, "%Y-%m-%d")
        until_list.append(until_plus_day)
        
    return since_list, until_list