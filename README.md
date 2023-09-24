# twitter-nlp
We present BotAegis; a Convulutional Neural Network (CNN) classifier which provides a score for the probability of Twitter accounts being bots, based on the content of a tweet. We have trained BotAegis on the [Internet Research Agency (IRA) Social Media Datast](https://www.kaggle.com/datasets/paultimothymooney/russian-political-influence-campaigns) of Russian 'troll farm' tweets from 2012 to 2018, mostly focused on the 2016 US election. We have applied it to English language tweets related to the initial months of the Ukraine-Russia conflict and also analyse the sentiment of the tweets as well as the probability of certain accounts of being bots. BotAegis can be trained on any dataset if one has a suitable dataset of known bot tweets (as well as known human tweets). 

Included in BotAegis is the option to call to [Botometer](https://botometer.osome.iu.edu/), a previous bot-detection tool which gives the probability of a twitter account being a bot based on user characteristics. We can therefore compare and contrast BotAegis which analyses the content of tweets directly and Botometer which analyses the stats of accounts. 

We also include a method of scraping tweets using [snscrape](https://github.com/JustAnotherArchivist/snscrape). You do not have to use this method and can scrape tweets in your own manner, but this method will return appropriately formated datasets. 

Finally we include a method of performing sentiment analysis over tweets and also a Latent Dirichlet Allocation (LDA) decomposition to investigate the topics present within tweets.

**Note that as of 30/06/2023 the scraping method using snscrape and the Botometer call are no longer functioning due to changes in the Twitter API** \
The core BotAegis code will remain functional, however, since it can work offline, only needing tweets scraped beforehand. You will therefore have to supply your own tweets. It appears that other versions of snscrape may be functional, although there may be limitations due to the change in Twitter API. 

Botometer currently is not working at all due to the API changes, although they aim to fix the issue, they will still be hampered by the API changes.

## Overview
It has been estimated that X % of Twitter accounts may be bots. Some of these accounts are harmless and do not attempt to hide the fact that they are bots and may serve useful functions such as making Y. However there appears to be a disturbingly high number of bot accounts which engage in nefarious behaviour, such as the accounts found to be tied to Russian interferene in the 2016 US election as mentioned above. Botted accounts or 'troll farm' tweets There has been academic interest in attempting to identify and tackle such accounts. However, despite the wealth of Data Science techniques and tools developed within recent years such as data collection and processing, Machine Learning, the increase in computational power and the continuing understanding of social media behaviours, there is no one reliable method for finding which accounts are legitimate accounts or bots. One major obstacle is the lack of data IS DATA   BOTOMETER USES ACCOUNT CHARACTERISTICS, BUT BASED ON A LIMITED SET OF DATA - IT REQUIRES A A LARGE AMOUNT OF MANUALLY COLLECTED USER DATA. IT IS ALSO FINELY-TUNED TO THAT DATA, ONLY ABLE TO ACCURATELY FIND BOTS SIMILAR TO THOSE IN THE TRAINING DATA. IT ALSO HAS LOTS OF BIASES - E.G. NEWLY CREATED ACCOUNTS ARE MUCH LIKELY TO BE PREDICTED AS BOTS. 

MANUAL COCLLECTION 

The field of bot detection is a constantly changing and evolving one, with those who make bots able to modify their approach should they need to. Furthermore, since every and all topics are talked about on social media, it (at present) seems an almost impossible task to detect all bots (unless companies such as Twitter are capable of doing further identification checks required to sign up). It can seem therefore that researchers are always two steps behind the bot-makers 

A
BOT AND HUMAN DATASETS

## Dependencies

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

If you want to scrape tweets or run Botometer you will also need a Twitter developer account to access the Twitter API. FINISH

RAPIDAPI

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
