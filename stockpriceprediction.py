from Tools import tools_v000 as tools
import os
from os.path import dirname
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
# Load the data

# -20 for the name of this project StockPricePrediction
save_path = dirname(__file__)[ : -20]
propertiesFolder_path = save_path + "Properties"

# Example of used
# user_text = tools.readProperty(propertiesFolder_path, 'StockPricePrediction', 'user_text=')

df = pd.read_csv('BTC-USD.csv')
# df = pd.read_csv('AMZN.csv')

df.set_index(pd.to_datetime(df['Date'].values))
# Give the index a name
df.index.name = 'Date' 

# Manipulate the data
# Create the target column
df['Price_Up'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Remove the date Column
df = df.drop(columns=['Date'])

# Show the data
print(df)

# Split the data set into a feature and a target data set
X = df.iloc[:, 0:df.shape[1]-1].values
Y = df.iloc[:, df.shape[1]-1].values

# Split the data again but this time into 80% training and 20% testing data sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Create and train the model (DecisionTreeClassifier)
tree = DecisionTreeClassifier().fit(X_train, Y_train)

# Show how well the model did on the test data set
print(tree.score(X_test, Y_test))

# Show the models predictions
tree_predictions = tree.predict(X_test)
print("Show the models predictions")
print(tree_predictions)

# Show the actual values
print("The actual values")
print(Y_test)


# path = tree.cost_complexity_pruning_path(X_train, Y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities



# fig, ax = plt.subplots()
# ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
# ax.set_xlabel("effective alpha")
# ax.set_ylabel("total impurity of leaves")
# ax.set_title("Total Impurity vs effective alpha for training set")



# plt.show()