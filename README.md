# Neural-Networks-Challenge

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

Using Pandas and scikit-learn’s StandardScaler(), we preprocess the dataset. Using TensorFlow, we design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. Then we create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras. As a final step for this model, we compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy using the test data and export our results into a HDF5 file.

In addition, we attempt to make this model more accurate (at least 75% accurate) by adjusting the input data to ensure that no variables or outliers are causing confusion in the model, such as:
-Dropping more or fewer columns.
-Creating more bins for rare occurrences in columns.
-Increasing or decreasing the number of values for each bin.
-Add more neurons to a hidden layer.
-Add more hidden layers.
-Use different activation functions for the hidden layers.
-Add or reduce the number of epochs to the training regimen.
