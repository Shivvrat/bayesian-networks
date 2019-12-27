import itertools
import math
import random
from operator import itemgetter

import import_data
import mixture_of_trees_using_EM
import numpy as np

import tree_bayesian_network


def creating_k_bags(train_dataset, k):
    num_of_examples = np.shape(train_dataset)[0]
    bags = dict()
    for each_k in range(k):
        data_in_this_bag = np.random.choice(num_of_examples, num_of_examples, replace=True)
        bags[each_k] = train_dataset[data_in_this_bag]
    return bags


def run_model(train_dataset, test_dataset, valid_dataset, k, r, num_of_iterations):
    log_likelihood_for_each_iteration = np.zeros((num_of_iterations, 1))
    for each_iteration in range(num_of_iterations):
        train_dataset_bags = creating_k_bags(train_dataset, k)
        mixture_probabilities = mixture_of_trees_using_EM.initialize_mixture_probabilities(k)
        for each_k in range(k):
            parameters = tree_bayesian_network.find_parameters(train_dataset_bags[each_k])
            mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset_bags[each_k],
                                                                                  parameters)
            zero_mutual_information_indices_feature_1 = np.reshape(np.random.choice(np.shape(mutual_information)[0], r),
                                                                   (r, 1))
            zero_mutual_information_indices_feature_2 = np.reshape(np.random.choice(np.shape(mutual_information)[0], r),
                                                                   (r, 1))
            zero_indices = np.concatenate(
                (zero_mutual_information_indices_feature_1, zero_mutual_information_indices_feature_2), axis=1)
            for each_row in zero_indices:
                mutual_information[each_row[0], each_row[1]] = 0
            mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
            edges_dict, edges = tree_bayesian_network.get_edges(mst)
            test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict, edges, test_dataset, parameters)
            log_likelihood_for_each_iteration[each_iteration] = log_likelihood_for_each_iteration[
                                                                    each_iteration] + test_log_likelihood + np.ma.log2(
                mixture_probabilities[each_k])
        log_likelihood_for_each_iteration[each_iteration] = (log_likelihood_for_each_iteration[each_iteration]) / float(
            k)
    log_likelihood_mean = np.mean(log_likelihood_for_each_iteration)
    log_likelihood_standard_deviation = np.std(log_likelihood_for_each_iteration)
    return log_likelihood_mean, log_likelihood_standard_deviation


def validation_of_model(dataset_name, num_of_iterations):
    train_dataset, test_dataset, valid_dataset = import_data.import_data(dataset_name)
    k = range(5, 21, 5)
    r = range(10, 1000, 100)
    best_k = 5
    best_r = 10
    best_log_likelihood = -math.inf
    for each in itertools.product(k, r):
        # Here I am testing on the validation dataset
        log_likelihood_mean, log_likelihood_standard_deviation = run_model(train_dataset, valid_dataset, test_dataset, each[0], each[1], num_of_iterations)
        if log_likelihood_mean > best_log_likelihood:
            best_k = each[0]
            best_r = each[1]
    log_likelihood_mean_final, log_likelihood_standard_deviation_final = run_model(train_dataset, test_dataset, valid_dataset, best_k, best_r, num_of_iterations)
    return log_likelihood_mean_final, log_likelihood_standard_deviation_final, best_k, best_r