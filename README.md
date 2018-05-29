# Regression


Requires Python version: 3.6.2

Python packages required:

a) pandas
b) sklearn
c) regex
d) numpy
e) matplotlib

command <pip install package_name> (eg. pip install pandas)

Steps to run the program:

1. Create a folder and place BikeRental.py, bikeRentalHourlyTrain.csv, bikeRentalHourlyTest.csv into it.
2. Open command prompt.
3. Go to the above folder where you have the python scripts and training/test datasets.
4. Run command <python BikeRental.py bikeRentalHourlyTrain.csv bikeRentalHourlyTest.csv>. Here we are passing <training dataset filename> as first argument and <test dataset filename> as second argument.
5. Output shows the Mean Square Error (MSE) obtained, using each of the techiniques(Neural Network, KNN, Linear REgression) on:-
a) Training data (5-fold cross validation)
b) Test data