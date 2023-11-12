# train.py

# Importing necessary libraries
import numpy as np
import nltk
from nltk.corpus import twitter_samples
# Importing custom functions from the 'functions.py' file
from functions import build_freqs, gradientDescent, extract_features, test_logistic_regression

# Download necessary datasets from NLTK
# This includes the twitter_samples and stopwords datasets
nltk.download('twitter_samples')
nltk.download('stopwords')

# Load Twitter samples for sentiment analysis
# all_positive_tweets and all_negative_tweets are lists of tweets, each containing positive and negative sentiments respectively
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Splitting the data into training and testing sets
# The first 4000 tweets from each sentiment are used for training, and the last 1000 tweets for testing
train_pos = all_positive_tweets[:4000]
train_neg = all_negative_tweets[:4000]
train_x = train_pos + train_neg  # Combining positive and negative training tweets
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)  # Creating labels for the training set

test_pos = all_positive_tweets[4000:]
test_neg = all_negative_tweets[4000:]
test_x = test_pos + test_neg  # Combining positive and negative testing tweets
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)  # Creating labels for the testing set

# Building a frequency dictionary for the training set
# The frequency dictionary maps each word in the training set with its corresponding frequency
freqs = build_freqs(train_x, train_y)

# Extracting features and stacking them into a matrix X
# Each row in X represents a tweet, and the columns represent the bias term and the positive and negative word counts
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

# Training the logistic regression model
# We initialize theta (parameters) to zeros, set a learning rate (alpha), and specify the number of iterations
J, theta = gradientDescent(X, train_y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")

# Saving the trained model parameters (theta) and the frequency dictionary (freqs) for later use
# These files will be loaded in the testing or plotting script
np.save('trained_theta.npy', theta)
np.save('freqs.npy', freqs)

# Optional: Calculating and printing the accuracy of the model on the test set
accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {accuracy:.4f}")
