# Internet-Connection-Request-Prediction
In this project, a neural networks are used to predict Internet connection request in mobile networks. The data is derived from [here](https://www.researchgate.net/publication/283432987_A_multi-source_dataset_of_urban_life_in_the_city_of_Milan_and_the_Province_of_Trentino/citations) which includes the Internet connection requests of Trento area for 60 days, while the area is devided to 6575 grid points and each grid point covers area of 235m x 235m. Further, each day interval is partitioned to 144 1ominutes time intervals. The goal is to predict the connection requests of mobile users using the previous connection request trends. 

This is a code package related to the following project. In this project, the goal is to predict the internet connection request using the data from previous timesteps. Each timestep is 10 minutes. A neural network is used for the prediction. The network contains input layer of size "train_window" which is the number of previous timesteps and 1 hidden layer with 200 neurons and output layer with just one neuron. Furthermore, for training, the 'adam' optimizer has been used which minimizes the loss which is mean squared error for this regression problem.

This repository contains the Python code required to reproduces all the numerical results.

## Content of Code Package
The package contains one Python file including Python code and one comma seperated file including one month of inetrnet connection request for Trentino city.
