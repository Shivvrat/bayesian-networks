import numpy as np


def fit(train_X):
    parameters = []
    for each_parameter in range(np.shape(train_X)[1]):
        value = (np.sum(train_X[:, each_parameter]) + 1) / (float(2 + np.shape(train_X)[0]))
        parameters.append(value)
    return parameters


def predict(test_X, parameters):
    log_likelihood = 0
    for each_column in range(np.shape(test_X)[1]):
        number_of_ones = np.sum(test_X[:, each_column])
        log_likelihood = log_likelihood + number_of_ones * np.math.log(parameters[each_column], 2) + (np.shape(test_X)[0] - number_of_ones) * np.math.log(1 - parameters[each_column], 2)
    return log_likelihood


"""
import import_data
import sys
import warnings
arguments = list(sys.argv)
try:
    data_set_name = str(arguments[1])
    algorithm_name = str(arguments[2])
except:
    print("You have given less arguments")
try:
    number_of_iterations = int(str(arguments[3]))
except:
    print(
        "You want to run Part 1 or Part 2, else you have provided wrong commands, please check. Please check the readme")
    

train_dataset, test_dataset, valid_dataset = import_data.import_data(data_set_name)
parameters = fit(train_dataset)
log_likelihood = predict(test_dataset, parameters)
print("The log likelihood is " + str(log_likelihood))

"""