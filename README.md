# twitter-nlp
We present BotAegis; a Convulutional Neural Network (CNN) classifier which provides a score for the probability of Twitter accounts being bots, based on the content of a tweet. We have trained BotAegis on the [Internet Research Agency (IRA) Social Media Datast](https://www.kaggle.com/datasets/paultimothymooney/russian-political-influence-campaigns) of Russian 'troll' farm tweets from 2012 to 2018, mostly focused on the 2016 US election. We have applied it to English language tweets related to the initial months of the Ukraine-Russia conflict and also analyse the sentiment of the tweets as well as the probability of certain accounts of being bots. BotAegis can be trained on any dataset if one has a suitable dataset of known bot tweets (as well as known human tweets). 

Included in BotAegis is the option to call to [Botometer](https://botometer.osome.iu.edu/), a previous bot-detection tool which gives the probability of a twitter account being a bot based on user characteristics. We can therefore compare and contrast BotAegis which analyses the content of tweets directly and Botometer which analyses the stats of accounts. 

We also include a method of scraping tweets using [snscrape](https://github.com/JustAnotherArchivist/snscrape). You do not have to use this method and can scrape tweets in your own manner, but this method will return appropriately formated datasets. 

Finally we include a method of performing sentiment analysis over tweets and also a Latent Dirichlet Allocation (LDA) decomposition to investigate the topics present within tweets.

**Note that as of 30/06/2023 the scraping method using snscrape and the Botometer call are no longer functioning due to changes in the Twitter API**\
The core BotAegis code will remain functional, however, since it can work offline, only needing tweets scraped beforehand. You will therefore have to supply your own tweets. It appears that other versions of snscrape may be functional, although there may be limitations due to the change in Twitter API. 

Botometer currently is not working at all due to the API changes, although they aim to fix the issue, they will still be hampered by the API changes.

## Overview

A
