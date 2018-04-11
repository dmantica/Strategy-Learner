# Strategy-Learner

Goal: Create a trading strategy for a stock using a random forest classifier and features generated from technical indicators


Contents

StrategyLearner.py: Creates random forest learner and uses it to generate trading strategy

RTLearner.py: A random decision tree learner class that contains methods for training and querying a decision tree

BagLearner.py: A random forest learner class that uses bootstrap aggregating to train and query a set of decision trees

indicators.py: Contains functions used to calculate different technical indicators (Bollinger Bands, momentum, midpoint) from historical stock prices

ManualStrategy.py: Implements a common type of manual trading strategy, used to compare to the machine learning generated trading strategy

marketsimcode.py: Used to simulate the performance of a portfolio created from a given trading strategy

util.py: Contains functions used to load stock data from csv and plot stock prices

experiment1.py: Test that compares the performance of the machine learning trading strategy to the manual trading strategy

experiment2.py: Test that examines how the machine learning trading strategy changes for different market impact values

report.pdf: Report detailing how the trading strategy is created and its performance
