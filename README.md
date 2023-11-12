# Twitter Sentiment Analysis using Logistic Regression

This project is a Twitter sentiment analysis tool that uses logistic regression to classify tweets as either positive or negative. The project is structured into three main components: function definitions, model training, and user interaction for testing the model or viewing the accuracy plot.

## Project Structure

- `functions.py`: Contains all the necessary functions for tweet processing, logistic regression, and prediction.
- `train.py`: Handles data loading, model training, and saves the trained model.
- `test_or_plot.py`: Offers an interactive way to test new tweets or show a plot of model accuracy.

## Getting Started

### Prerequisites

Before running this project, you need to install Python and the necessary libraries. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Installation

* 1. Clone the repo:
```
git clone https://github.com/mouraffa/Sentimental_Analysis_LogRegClassification.git
```
*2. Install the required packages
```
pip install -r requirements.txt
```
### Usage

Run `train.py` to train the model and save the parameters:

```
python train.py
```
After training the model, you can either test it with your own tweets or view the accuracy plot by running:
```
python test_or_plot.py
```
### How it Works

* `train.py` downloads tweet data, processes it, and uses it to train a logistic regression model.
* `test_or_plot.py` allows you to input a custom tweet to get a sentiment prediction or to view a plot of the model's training accuracy.

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

* 1. Fork the Project
* 2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
* 3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
* 4. Push to the Branch (`git push origin feature/AmazingFeature`)
* 5. Open a Pull Request

## Contact
[LinkedIn](https://www.linkedin.com/in/youssef-mouraffa-316663201/)
[Email](mouraffayoussef@gmail.com)
