# twitter-nlp
We present BotAegis; a Convulutional Neural Network (CNN) classifier which provides a score for the probability of Twitter accounts being bots, based on the content of a tweet. We have trained BotAegis on  [FiveThirtyEight's dataset](https://www.kaggle.com/datasets/paultimothymooney/russian-political-influence-campaigns) of Russian Internet Research Agency (IRA) Social Media 'troll farm' tweets from 2012 to 2018, mostly focused on the 2016 US election. We have applied it to English language tweets related to the initial months of the Ukraine-Russia conflict and also analyse the sentiment of the tweets as well as the probability of certain accounts of being bots. BotAegis can be trained on any dataset if one has a suitable dataset of known bot tweets (as well as known human tweets). 

Included in BotAegis is the option to call to [Botometer](https://botometer.osome.iu.edu/), a previous bot-detection tool which gives the probability of a twitter account being a bot based on user characteristics. We can therefore compare and contrast BotAegis which analyses the content of tweets directly and Botometer which analyses the stats of accounts. 

We also include a method of scraping tweets using [snscrape](https://github.com/JustAnotherArchivist/snscrape). You do not have to use this method and can scrape tweets in your own manner, but this method will return appropriately formated datasets. 

Finally we include a method of performing sentiment analysis over tweets and also a Latent Dirichlet Allocation (LDA) decomposition to investigate the topics present within tweets.

**Note that as of 30/06/2023 the scraping method using snscrape and the Botometer call are no longer functioning due to changes in the Twitter API** \
The core BotAegis code will remain functional, however, since it can work offline, only needing tweets scraped beforehand. You will therefore have to supply your own tweets. It appears that other versions of snscrape may be functional, although there may be limitations due to the change in Twitter API. 

Botometer currently is not working at all due to the API changes, although they aim to fix the issue, they will still be hampered by the API changes.

## Overview
It has been estimated that XXX % of Twitter accounts may be bots. Some of these accounts are harmless and do not attempt to hide the fact that they are bots and may serve useful functions such as making YYY. However there appears to be a disturbingly high number of bot accounts which engage in nefarious behaviour, such as the accounts found to be tied to Russian interferene in the 2016 US election as mentioned above. Botted accounts or 'troll farm' tweets There has been academic interest in attempting to identify and tackle such accounts. However, despite the wealth of Data Science techniques and tools developed within recent years such as data collection and processing, Machine Learning, the increase in computational power and the continuing understanding of social media behaviours, there is no one reliable method for finding which accounts are legitimate accounts or bots. One major obstacle is the lack of data on bot accounts. Manual serveys such as tht on the IRA Social Media Dataset mentioned above are of vital importance and have been successful in identifying at some bots. However they are costly in time and resources to run and can only capture a relatively small percentage of bots. We wish, therefore, to develop automatic detection software that can be used to detect new instances of bots.

This is what Botometer was also intended for - Botometer looks at the characteristics of user's accounts and looks for features which may indicate that it could be a bot and uses a trained network to predict if new accounts could be bots. An inescapable fact of Machine Learning methods such as Botometer and now BotAegis, is that they are really only as good as the data that they are trained on. Botometer has been trained on a dataset of known bot accounts (as well as verified human accounts) and can predict whether an account is likely to be a certain type of bot such as political 'astroturfing' bots, bots purchased to increase follower counts, spam bots etc. (see [here](https://rapidapi.com/OSoMe/api/botometer-pro/details) for details). However some of the datasets used in training are quite limited, for example the astroturf dataset has only around 500 bots in it. Whilst it is great that these datasets exist, and without the work of the people making them then we would have no data to work with, they are not ideal for training large neural networks. 

We found that Botometer often overfits to the type of accounts that it has trained on, giving a significant number of false positives and false negtives, since it does not know how to account for ore varied real-world examples. Furthermore it also comes with other biases, such as giving a much higher chance of predicting an account to be a bot if it was created recently. This is not intended as a slight upon Botometer - it is a remarkable tool and works very well in certain cases, however it does face some shortcomings due to its limited training sample size and variance. This is also a problem that we encounter with BotAegis - we will also overfit to the samples which we have trained upon - there just does not exist as of yet data large, reliable and varied enough for truly accurate bot detection. However we are certainly able to detect certain accounts, which appear to bots - in our analysis we find around 800 potentially bot or bot-like accounts - but we know that there will be bots that we will miss unless we can train using better datasets.

About botaegis... 
Instead of looking at account user characteristics like Botometer does, BotAegis analyses instead the content of tweets directly. In this Natural Language Processing (NLP) approach, we can skip 


Ukraine

WE also overfit

The field of bot detection is a constantly changing and evolving one, with those who make bots able to modify their approach should they need to. Furthermore, since every and all topics are talked about on social media, it (at present) seems an almost impossible task to detect all bots (unless companies such as Twitter are capable of doing further identification checks required to sign up). It can seem therefore that researchers are always two steps behind the bot-makers 

A
BOT AND HUMAN DATASETS

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
- `train_botaegis.py`: traings the botaegis CNN classifier on the scraped tweets (or your own tweets dataset). This uses the `TrainBotAegis` class within `botaegis`. BOT AND HUMAN DATASETS

### tweetsearcher
SCRAPING!

### botaegis
The `botaegis` folder, which contains five important classes: 
- `TrainBotAegis` contained within `train_botaegis_classifier.py`
- `RunBotAegis` contained within `run_botaegis_classifier.py`
- `RunBotometer` contained within `run_botometer_classifier.py`
- `GetSentiment` and `LDA_Decomposition` contained within `run_sentiment_analysis.py`

`TrainBotAegis` as the name suggests, is a class which contains all the methods used for training the BotAegis classifier, including reading in datasets and preparing them for training (including tokenization, padding and train-test-splitting). It will also build the model which is specified (see **models**)


### models

### configs 

### Other folders
