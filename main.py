import sys
import warnings

import EM_using_random_forest
import import_data
import independent_bayesian_network
import mixture_of_trees_using_EM
import tree_bayesian_network

warnings.filterwarnings("ignore")
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


def main():
    """
    This is the main function which is used to run all the algorithms
    :return:
    """
    try:
        if algorithm_name == '-ibn':
            # This is for multi-nomial naive bayes
            train_dataset, test_dataset, valid_dataset = import_data.import_data(data_set_name)
            parameters = independent_bayesian_network.fit(train_dataset)
            log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
            print("The log likelihood is " + str(log_likelihood))
        elif algorithm_name == '-tbn':
            # This is for discrete naive bayes
            train_dataset, test_dataset, valid_dataset = import_data.import_data(data_set_name)
            parameters = tree_bayesian_network.find_parameters(train_dataset)
            mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
            mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
            edges_dict, edges = tree_bayesian_network.get_edges(mst)
            test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict, edges, test_dataset, parameters)
            print("The log likelihood is " + str(test_log_likelihood))
        elif algorithm_name == '-mtem':
            # This is for the MCAP algorithm
            mean, standard_deviation = mixture_of_trees_using_EM.validation_of_model(data_set_name,
                                                                                     number_of_iterations)
            print("The log likelihood mean and standard deviation are ", str(mean), str(standard_deviation))

        elif algorithm_name == '-mtemrf':
            # This is for the SGD classifier from sklearn
            log_likelihood_mean, log_likelihood_standard_deviation = EM_using_random_forest.validation_of_model(
                data_set_name, number_of_iterations)
            print("The log likelihood mean and standard deviation are ", str(log_likelihood_mean),
                  str(log_likelihood_standard_deviation))

        else:
            print('You have entered wrong value for the algorithm, please check in readme')
            return
    except:
        print('Something went wrong please check the command line parameters again from the readme or check if the '
              'dataset folder is in the right place or not')


if __name__ == "__main__":
    main()
