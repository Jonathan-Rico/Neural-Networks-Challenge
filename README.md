# Neural-Networks-Deep-Learning-Challenge

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

Results

In our first model, we chose the variable IS_SUCCESSFUL as the target variable. The EIN and NAME column were dropped, application types which had counts above 600 and
classifications which had counts above 800 were dropped from the dataframe (to remove outliers) and reclassified as `other`. The categorical data is then converted to numeric values (binary) in order to run it through the model.
We run the first sequential model using 2 hidden layers (1st layer = 100 neurons, 2nd layer = 50 neurons) using a rectified linear unit function, while the output layer (1 neuron) used a sigmoid function.

![image](https://user-images.githubusercontent.com/6768169/226886322-d0d88e56-c3d7-43e8-ba0c-f08676629386.png)

Our first model's results were:

268/268 - 0s - loss: 0.5624 - accuracy: 0.7287 - 251ms/epoch - 936us/step
Loss: 0.562398374080658, Accuracy: 0.7287463545799255.

Optimization:

To optimize the model , I chose to drop only the EIN column form the dataframe, and chose to explore data with whose NAME counts were below 5, as well as classification counts that were lower than 800, and reclassify the rest of the data as `other` (chose to leave in all application types). By looking at more categories of the data, while still filtering out outliers in the dataset while looking at more features, we can hope to get a better representation of our dataset when we run our data through the model.
This model ran using 4 hidden layers, with the first one (100 neurons) starting as a rectified linear unit function and the other three layers (30 neurons, 10 neurons and 5 neurons) using a sigmoid function. The output layer (1 neuron) also used a sigmoid function. By using more layers (with decreasing neuron counts per layer), I hoped to increase the accuracy of the model.
to the desired accuracy level (>75%)

![image](https://user-images.githubusercontent.com/6768169/226889167-77b45903-9427-4d75-81c0-55a03fa9317b.png)

The results for this model were:
268/268 - 0s - loss: 0.4585 - accuracy: 0.7914 - 241ms/epoch - 900us/step
Loss: 0.4584789574146271, Accuracy: 0.7913702726364136

This model was successful at reaching the desired accuracy of 75% or greater accuracy. From this result, we can assume that our first model's scope (looking at the features) omitted an important column for analysis, as well as lacked enough parameters (layers+neurons) to effectively make predictions on our dataset. 

The second model shows us that an applicant has a 79% chance of being successful if they met both the following criteria:

--Contained the classifications C1000,C1200,C2000,C2100,C3000

--Their name appeared more than 4 times in the dataset (applied more than 4 times)

A recommendation I would make to help solve this classification problem is to use a Random Forest model, which would help us calculate the importance of each feature in our dataset, and make better adjustments on our deep learning model based on our results from the Random Forest model.
