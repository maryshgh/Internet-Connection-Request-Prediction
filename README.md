# Internet-Connection-Request-Prediction
In this project, a neural networks are used to predict Internet connection request in mobile networks.

This is a code package related to the following project. In this project, the goal is to predict the internet connection request using the data from previous timesteps. Each timestep is 10 minutes. A neural network is used for the prediction. The network contains input layer of size "train_window" which is the number of previous timesteps and 1 hidden layer with 200 neurons and output layer with just one neuron. Furthermore, for training, the 'adam' optimizer has been used which minimizes the loss which is mean squared error for this regression problem.

This repository contains the Python code required to reproduces all the numerical results.

## Content of Code Package
The package contains one Python file including Python code and one comma seperated file including one month of inetrnet connection request for Trentino city.
