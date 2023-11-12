# test_or_plot.py

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from functions import predict_tweet

# Load the trained model parameters and frequency dictionary
# These files were saved by the train.py script
theta = np.load('trained_theta.npy')
freqs = np.load('freqs.npy', allow_pickle=True).item()

# Asking the user for their choice
# Users can choose between testing a new tweet or displaying the accuracy plot
user_choice = input("Enter 'plot' to show the accuracy plot, or 'tweet' to test your own tweet: ").strip().lower()

if user_choice == 'plot':
    # Code for plotting model accuracy
    # You need to replace the following example accuracies with the actual accuracies from your model
    accuracies = [0.8, 0.82, 0.85, 0.87, 0.9]  # Example accuracies, replace with actual values

    # Plotting the accuracies
    plt.plot(accuracies)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Iterations')
    plt.show()

elif user_choice == 'tweet':
    # Code for testing a custom tweet
    # Users can enter their own tweet, and the script will use the model to predict its sentiment
    custom_tweet = input("Enter your tweet: ")
    y_hat = predict_tweet(custom_tweet, freqs, theta)
    
    # Displaying the sentiment prediction
    if y_hat > 0.5:
        print('Positive sentiment')
    else: 
        print('Negative sentiment')

else:
    # Handling invalid input
    print("Invalid choice. Please enter 'plot' or 'tweet'.")
