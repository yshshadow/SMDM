# Sentiment Analysis of Tweets and Comments
Analysis President Trump’s tweets and comments to those tweets by Naive Bayes and K-Means algorithm. The result of analysis is shown by charts drawn by Bokeh and Matplotlib.

## Requirement
Python3
Pandas https://pandas.pydata.org/
Numpy http://www.numpy.org/
Bokeh https://bokeh.pydata.org/ 
NLTK https://www.nltk.org/
Scikit-learn https://scikit-learn.org/
Matplotlib https://matplotlib.org/
wordcloud https://github.com/amueller/word_cloud
Tweepy http://www.tweepy.org/

## Data Set
<b>data/Tweets.csv</b> Trump's tweets.
<b>data/Comments.csv</b> Comments on Trump's tweets.

## Other data Set
Sentiment 140 http://help.sentiment140.com/for-students

## Codes
<b>Main.ipynb</b> Main program to do sentiment analysis, clustering and result visualization.
<b>TwitterSpider.py</b> Main program to collect data from Twitter.
<b>TwitterCluster.py</b> Implement tweets clustering.
<b>TwitterSentimentAnalyzer</b> Implement sentiment analysis on tweets.

## Other files or subfolders
<b>config.json</b> Configuration JSON file to store Twitter API OAuth information.
<b>model/</b> Save Naive Bayes and K-Means models
<b>result/</b> Save results.

## How to use
To collect data from Twitter, fill your Twitter API key in config.json and run TwitterSpider.py.
To run clustering and sentiment analysis, remove saved models in model subfolder and run Main.ipynb.
To see the result, run Main.ipynb directly.