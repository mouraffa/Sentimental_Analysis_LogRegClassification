import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Function Definitions

# Processing Tweet Function
def process_tweet(tweet):
    """
    Process tweet function: Cleans and tokenizes tweets.

    Arguments:
    tweet -- a string containing a tweet

    Returns:
    tweets_clean -- a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = [stemmer.stem(word) for word in tweet_tokens if word not in stopwords_english and word not in string.punctuation]
    return tweets_clean

# Building Frequency Dictionary Function
def build_freqs(tweets, ys):
    """
    Build frequencies of words in tweets with their sentiment label.

    Arguments:
    tweets -- a list of tweets
    ys -- an m x 1 array with the sentiment label of each tweet (either 0 or 1)

    Returns:
    freqs -- a dictionary mapping each (word, sentiment) pair to its frequency
    """
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs

# Sigmoid Function
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    h -- sigmoid(z)
    """
    h = 1 / (1 + np.exp(-z))
    return h

# Gradient Descent Function
def gradientDescent(x, y, theta, alpha, num_iters):
    """
    Perform gradient descent to learn theta.

    Arguments:
    x -- matrix of features which is (m,n+1)
    y -- corresponding labels of the input matrix x, dimensions (m,1)
    theta -- weight vector of dimension (n+1,1)
    alpha -- learning rate
    num_iters -- number of iterations to run gradient descent

    Returns:
    J -- the final cost
    theta -- final weight vector
    """
    m = len(x)
    for i in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        theta = theta - (alpha/m) * np.dot(x.T, (h - y))
    return float(J), theta

# Extract Features Function
def extract_features(tweet, freqs):
    """
    Extract features from a single tweet.

    Arguments:
    tweet -- a string containing a tweet
    freqs -- a dictionary with the frequency of each pair (or tuple)

    Returns:
    x -- a feature vector of dimension (1,3)
    """
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3)) 
    x[0,0] = 1 
    for word in word_l:
        x[0,1] += freqs.get((word, 1.), 0)
        x[0,2] += freqs.get((word, 0.), 0)
    return x

# Predict Tweet Function
def predict_tweet(tweet, freqs, theta):
    """
    Predict whether a tweet is positive or negative.

    Arguments:
    tweet -- a string
    freqs -- a dictionary with the frequency of each pair (or tuple)
    theta -- (3,1) vector of weights

    Returns:
    y_pred -- the probability of a tweet being positive or negative
    """
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

# Test Logistic Regression Function
def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Test the logistic regression model.

    Arguments:
    test_x -- a list of tweets
    test_y -- (m, 1) vector with the corresponding labels for the list of tweets
    freqs -- a dictionary with the frequency of each pair (or tuple)
    theta -- weight vector of dimension (3, 1)

    Returns:
    accuracy -- (# of tweets classified correctly) / (total # of tweets)
    """
    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        y_hat.append(1.0 if y_pred > 0.5 else 0.0)
    accuracy = np.mean(np.array(y_hat) == np.squeeze(test_y))
    return accuracy
