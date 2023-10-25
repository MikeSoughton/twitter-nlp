# twitter-nlp
We present BotAegis; a Convulutional Neural Network (CNN) classifier which provides a score for the probability of Twitter accounts being bots, based on the content of a tweet. We have trained BotAegis on  [FiveThirtyEight's dataset](https://www.kaggle.com/datasets/paultimothymooney/russian-political-influence-campaigns) of Russian Internet Research Agency (IRA) Social Media 'troll farm' tweets from 2012 to 2018, mostly focused on the 2016 US election. We have applied it to English language tweets related to the initial months of the Ukraine-Russia conflict and also analyse the sentiment of the tweets as well as the probability of certain accounts of being bots. BotAegis can be trained on any dataset if one has a suitable dataset of known bot tweets (as well as known human tweets). 

Included in BotAegis is the option to call to [Botometer](https://botometer.osome.iu.edu/), a previous bot-detection tool which gives the probability of a twitter account being a bot based on user characteristics. We can therefore compare and contrast BotAegis which analyses the content of tweets directly and Botometer which analyses the stats of accounts. 

We also include a method of scraping tweets using [snscrape](https://github.com/JustAnotherArchivist/snscrape). You do not have to use this method and can scrape tweets in your own manner, but this method will return appropriately formated datasets. 

Finally we include a method of performing sentiment analysis over tweets and also a Latent Dirichlet Allocation (LDA) decomposition to investigate the topics present within tweets.

**Note that as of 30/06/2023 the scraping method using snscrape and the Botometer call are no longer functioning due to changes in the Twitter API** \
The core BotAegis code will remain functional, however, since it can work offline, only needing tweets scraped beforehand. You will therefore have to supply your own tweets. It appears that other versions of snscrape may be functional, although there may be limitations due to the change in Twitter API. 

Botometer currently is not working at all due to the API changes, although they aim to fix the issue, they will still be hampered by the API changes.

## Overview
It has been estimated that up to 15 % of Twitter accounts may be bots. Some of these accounts are harmless and do not attempt to hide the fact that they are bots and may serve useful and legitimate functions. However there appears to be a disturbingly high number of bot accounts which engage in nefarious behaviour, such as the accounts found to be tied to Russian interferene in the 2016 US election as mentioned above. Botted accounts or 'troll farm' tweets There has been academic interest in attempting to identify and tackle such accounts. However, despite the wealth of Data Science techniques and tools developed within recent years such as data collection and processing, Machine Learning, the increase in computational power and the continuing understanding of social media behaviours, there is no one reliable method for finding which accounts are legitimate accounts or bots. One major obstacle is the lack of data on bot accounts. Manual serveys such as tht on the IRA Social Media Dataset mentioned above are of vital importance and have been successful in identifying at some bots. However they are costly in time and resources to run and can only capture a relatively small percentage of bots. We wish, therefore, to develop automatic detection software that can be used to detect new instances of bots.

This is what Botometer was also intended for - Botometer looks at the characteristics of user's accounts and looks for features which may indicate that it could be a bot and uses a trained network to predict if new accounts could be bots. An inescapable fact of Machine Learning methods such as Botometer and now BotAegis, is that they are really only as good as the data that they are trained on. Botometer has been trained on a dataset of known bot accounts (as well as verified human accounts) and can predict whether an account is likely to be a certain type of bot such as political 'astroturfing' bots, bots purchased to increase follower counts, spam bots etc. (see [here](https://rapidapi.com/OSoMe/api/botometer-pro/details) for details). However some of the datasets used in training are quite limited, for example the astroturf dataset has only around 500 bots in it. Whilst it is great that these datasets exist, and without the work of the people making them then we would have no data to work with, they are not ideal for training large neural networks. 

We found that Botometer often overfits to the type of accounts that it has trained on, giving a significant number of false positives and false negtives, since it does not know how to account for ore varied real-world examples. Furthermore it also comes with other biases, such as giving a much higher chance of predicting an account to be a bot if it was created recently. This is not intended as a slight upon Botometer - it is a remarkable tool and works very well in certain cases, however it does face some shortcomings due to its limited training sample size and variance. This is also a problem that we encounter with BotAegis - we will also overfit to the samples which we have trained upon - there just does not exist as of yet data large, reliable and varied enough for truly accurate bot detection. However we are certainly able to detect certain accounts, which appear to bots - in our analysis we find around 800 potentially bot or bot-like accounts - but we know that there will be bots that we will miss unless we can train using better datasets.

Instead of looking at account user characteristics like Botometer does, BotAegis analyses instead the content of tweets directly. In this Natural Language Processing (NLP) approach, we can uncover similarities within the speech patterns of bot farm accounts. BotAegis is a 1D CNN, with 14 convolutional layers using n-grams and skip-grams, which aims to predict whether a tokenized Tweet belongs to the human class or the bot class. In that sense it is solving a simple classification problem, just with a large vocabulary and speech patterns to be learnt. We trained initially on bot tweets from the IRA dataset alongside human tweets from the [verified-2019 dataset](https://botometer.osome.iu.edu/bot-repository/datasets.html) before supplementing them with the [TwiBot-20 dataset](https://botometer.osome.iu.edu/bot-repository/datasets.html) and verified Twitter users from this [Reddit dataset](https://www.reddit.com/r/datasets/comments/8s6nqz/all_verified_twitter_users_100_complete_in_ndjson/).     

Once trained, we applied BotAegis to tweets mentioning 'Ukraine' and 'Russia' (in English language) after the advent of the invasion, looking to investigate the operation of bots within this sphere. We find a number of accounts which are predicted to be bots based on their tweets. Note that this does mean that these accounts *are* bots, just that their speech patterns are similar to that of the bots which we trained on. There is always the possibility of obtaining false positives - genuine accounts which are flagged as being bots. This is always a concern when using Machine Learning and ethical use of it should be considered alongside traditional authentication procedures so that genuine people are not penalised. We performed some manual checks on accounts in an attempt to verify if an account is genuinely a bot or human, for example inspecting their post history, whether they use their own profile picture or a stock image taken from the web, the lack of friends/followers etc., however we did not have the resources to check this properly. We are confident that at least a significant number of flagged tweets appear to come from bot accounts. We further compared our results alongside Botometers - whilst both Botometer and BotAegis may overfit in different ways, if an account scores highly for both of them then it is definetely worthy of suspicion. We also noted that that whilst some of the accounts predicted to be bots appear to be harmless and are likely false positives, we also find other accounts predicted to be bots, whilst their user characteristics suggest that they are genuine people, are engaging in speech similar to that of bots. That is not a bad thing per se, bots after all are attempting to mimic humans, but there is a fine line between hate speech commited by bots and hate speech commited by humans, which is something worth looking into in the future.

One thing that we do notice is that we are much more likely to predict a tweet to be a bot tweet if it mentions things similar to topics within the training dataset, for example if they mention "Trump" or "Biden", though in the context of Ukraine. We will leave the analysis for elsewhere, but this can be an insight into how the bots operate. We also perform sentiment analysis of the tweets to track the sentiment over time and compare and contrast the sentiment expressed by bots vs humans (bots consistently are more negative). We also investigate the topics talked about by both humans and bots through a Latent Dirichlet Allocation (LDA) decomposition as well as a breakdown of the sentiment by topic.

The field of bot detection is a constantly changing and evolving one, with those who make bots able to modify their approach should they need to. Furthermore, since every and all topics are talked about on social media, it (at present) seems an almost impossible task to detect all bots (unless companies such as Twitter are capable of doing further identification checks required to sign up). It can seem therefore that researchers are always two steps behind the bot-makers. BotAegis makes steps towards a workable bot-detection system which can be used to identify suspicious accounts. We are aware however, that more data is required for truly accurate detection and that training data should be up-to-date and relevant for the topics in question. This is an challenge that will have to be tackled through collaboration through academia, tech companies and governments.

## Dependencies and Setup

The code is run in `python3.8.5`. The following packages are required:

```
keras==2.4.3
numpy==1.21.5
pandas==1.4.2
matplotlib==3.5.1
tensorflow==2.4.1
scikit-learn==1.0.2
tqdm==4.64.0
vadersentiment==3.3.2
nltk==3.7
ipykernel==6.9.1
seaborn==0.11.2
snscrape==0.4.3
```

These can be installed manually or via the conda yaml file using

```
conda env create --name <env name> -f environment.yml
```

If you wish to run Botometer you will need a Twitter developer account to access the Twitter API. You will have to put your API key, API secret, access token and access secret in a file inside `configs`, making sure that the filename matches that in `run_botometer_config.json` (see `config_description.txt` for details). 

You will also need to a RapidAPI account to interface with Botometer. Instructions can be found in [Botometer's API documentation](https://botometer.osome.iu.edu/api) which instruct you to sign up to their [RapidAPI endpoint](https://rapidapi.com/OSoMe/api/botometer-pro). There, after choosing a plan, you can get your RapidAPI key which can be downloaded and placed in `configs` to be read, making sure that the filename matches that in `run_botometer_config.json` (see `config_description.txt` for details).

Twitter API keys should be stored within a json file named `twitter_keys_config.json`, with format
```
{
"API_key": "XXX",
"API_secret": "XXX",
"access_token": "XXX",
"access_secret": "XXX"
}
```

RapidAPI keys should be stored within a json file named `rapidapi_keys_config.json`, with format
```
{
"API_key": "XXX"
}
```

## Code layout
The main bulk of the code is found within in `tweetsearcher` and `botaegis` package directories. The former contains the classes used for scraping tweets and the latter contains the classes used for training and running BotAegis as well as for running Botometer and performing sentiment and topic analysis. The scripts in the main directory will instantiate the classes within the above package directories and are the only scripts that need be run. The scripts all call arguments from config files within `configs`. Inside `configs` is a text file describing the meanings and uses of the parameters within the config files. The scripts themselves are:
- `scrape_tweets.py`: scrapes tweets using snscrape. This uses the `tweetsearcher` package directory to do so. There are two modes through which tweets can be scraped, either by keywords contained within tweets or by the user id. We predominately scraped using keywords for this project. One can search for tweets between a certain date range, and get a specified number of tweets from each day containing certain keywords. This script will also clean the tweets and filter out accounts not used by individuals (i.e. organisational accounts or 'good' bot accounts). The script will output csv files in a newly created directory named `data` generated from saving dataframes containing the tweets.
- `train_botaegis.py`: traings the BotAegis CNN classifier on the scraped training tweets (or your own tweets dataset). This uses the `TrainBotAegis` class within `botaegis`, training on data which should be stored within a directory inside `data`. The data is cleaned and tokenised before being fed into the classifier. Saved trained model hdf5 files will be saved within `models/checkpoints` as well as a tokenizer json file for use in running the classifier.
- `run_botaegis.py`: runs the BotAegis CNN classifier on new scraped tweets. This uses the `RunBotAegis` class within `botaegis`, running on data stored within a different directory inside `data`. The tweets are prepared in a similar way as were the training tweets, then run through the trained model, which is loaded in alongside the tokenizer saved previously. A new dataframe with the BotAegis scores for the tweets is then saved in a specified location within `data`.
- `run_botometer.py`: calls the Botometer API to return Botometer scores for user accounts. This uses the `RunBotometer` class within `botaegis` (we lumped the Botometer call in alongside BotAegis since they compliment each other nicely). It takes in the same tweet dataset as `run_botaegis` and calls Botometer to return scores for the accounts listed (it will automatically handle duplicate tweets from the same account since Botometer checks by account, not tweet). Results can be outputted in the same location as the BotAegis results.
- `run_sentiment.py`: runs sentiment analysis and LDA analysis over the tweets. This uses the `GetSentiment` class and `LDA_Decomposition` classes within `botaegis`. Either the sentiment analysis or LDA decomposition can be run seperately or both can be run together. The inputs should be the tweet csv files outputted after running BotAegis and Botometer (the sentiment analysis code does not require bot scores, but the LDA decompostion takes the bot scores to perform further analysis. The sentiment analysis code will output a new file with the sentiment scores and the LDA analysis code will output a new file with conversation topic numbers, as well as html files showing a visualisation of the topics.

### tweetsearcher
The `tweetsearcher` package directory contains two important classes:
-  `GetTweetsByKeyword` contained within `scrape_by_keyword.py`
-  `GetTweetsById` contained within `scrape_by_id.py`

`GetTweetsByKeyword` as the name suggests, is a class which will scrape for Tweets containing certain keyword(s). Given a list of keywords, a start date, end date, number of tweets per day and some other parameters such as language and output file location, it runs snscrape to output a tweets dataframe csv file containing the tweets as well as their corresponding metadata. It also will clean out organisational tweets and tweets from accounts with non-human names.

`GetTweetsById`, unsurprisingly, is a class which will scrape for Tweets from specific user ids. Given a list of user ids, max number of tweets per user and a few other input parameters, it runs snscrape to output a tweets dataframe csv file containing the tweets as well as their corresponding metadata. Since we only used this code to download tweets from already known human accounts from the Reddit dataset, this class does not need to filter out organisational tweets and tweets from accounts with non-human names.

### botaegis
The `botaegis` package directory contains five important classes: 
- `TrainBotAegis` contained within `train_botaegis_classifier.py`
- `RunBotAegis` contained within `run_botaegis_classifier.py`
- `RunBotometer` contained within `run_botometer_classifier.py`
- `GetSentiment` and `LDA_Decomposition` contained within `run_sentiment_analysis.py`

`TrainBotAegis` as the name suggests, is a class which contains all the methods used for training the BotAegis classifier, including reading in datasets (there are options for either the IRA or the Twibot bot datasets as well as the human dataset, allowing for easy setup of different data formats) and preparing them for training (including tokenization, padding and train-test-splitting). It will also build the chosen model (see [**models**](#models)) before training with it and finally testing. The trained model weights as well as the tokenizer are saved within `models`. 

`RunBotAegis` allows for running the trained BotAegis model on your desired Tweet dataset. Within the class there are methods for reading in and preparing this data, as well as the model, trained model weights and tokenizer, before running the model and saving the results as a new tweet dataset containing the BotAegis bot scores as well as all the previous data. 


### models

### configs 

### Other folders
