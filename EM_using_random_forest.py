import itertools
import math
from operator import itemgetter
from random import random
import import_data
import numpy as np
import networkx
from numpy.ma import log2

import tree_bayesian_network
import import_data


def initialize_trees(train_dataset):
    graph = initialize_tree_parameters(train_dataset)
    max_spanning_tree = tree_bayesian_network.find_max_spanning_tree(graph)
    return max_spanning_tree


def initialize_tree_parameters(train_dataset):
    num_of_features = np.shape(train_dataset)[1]
    graph = networkx.fast_gnp_random_graph(n=num_of_features, p=0.3)
    array = np.zeros((num_of_features, num_of_features))
    for each_vertex in graph:
        for each_opposite_vertex in graph[each_vertex]:
            array[each_vertex][each_opposite_vertex] = 1
    return array


def initialize_k_trees(train_dataset, k):
    k_trees = dict()
    for each_tree in range(k):
        k_trees[each_tree] = initialize_trees(train_dataset)
    return k_trees


def initialize_mixture_probabilities(k):
    mixture_probab = [1 / k] * k
    return np.array(mixture_probab)


def initialize_joint_probability_distribution(k, train_dataset, k_trees):
    joint_probability_distribution = dict()
    for each_k in range(k):
        joint_probability_distribution[each_k] = dict()
        for each_attribute_1 in range(np.shape(train_dataset)[1]):
            joint_probability_distribution[each_k][each_attribute_1] = dict()
            for each_attribute_2 in range(np.shape(train_dataset)[1]):
                if k_trees[each_k][each_attribute_1][each_attribute_2] != 0:
                    joint_probability_distribution[each_k][each_attribute_1][each_attribute_2] = np.random.random(
                        (2, 2))
            # -1 is for self probability
            joint_probability_distribution[each_k][each_attribute_1][-1] = random()
    return joint_probability_distribution


def compute_mutual_information(train_dataset, complete_data, cluster_number):
    probability_distribution = dict()
    probability_distribution_each_parameter = dict()
    num_of_examples = np.shape(train_dataset)[0]
    num_of_features = np.shape(train_dataset)[1]
    mutual_information = np.zeros((num_of_features, num_of_features))
    denominator = np.sum(complete_data[:, cluster_number]) + 2
    for each_parameter in range(num_of_features):
        count = 2
        for each_example in range(num_of_examples):
            if train_dataset[each_example][each_parameter] == 1:
                count = count + complete_data[each_example][cluster_number]
        probability_distribution_each_parameter[each_parameter] = count / (float(denominator))
    for count1 in range(num_of_features):
        probability_distribution[count1] = dict()
        for count2 in range(num_of_features):
            each_case = np.zeros((2, 2))
            array_temp = train_dataset[:, (count1, count2)]
            for each_tuple in array_temp:
                if np.array_equal(each_tuple, [0, 0]):
                    each_case[0, 0] += 1
                if np.array_equal(each_tuple, [1, 0]):
                    each_case[1, 0] += 1
                if np.array_equal(each_tuple, [0, 1]):
                    each_case[0, 1] += 1
                if np.array_equal(each_tuple, [1, 1]):
                    each_case[1, 1] += 1
            for each_row in range(np.shape(each_case)[0]):
                for each_column in range(np.shape(each_case)[1]):
                    p_uv = (each_case[each_row][each_column] + 1) / float(num_of_examples + 2)
                    if each_row == 0:
                        p_u = 1 - probability_distribution_each_parameter[count1]
                    else:
                        p_u = probability_distribution_each_parameter[count1]
                    if each_column == 0:
                        p_v = 1 - probability_distribution_each_parameter[count2]
                    else:
                        p_v = probability_distribution_each_parameter[count2]
                    try:
                        mutual_information[count1][count2] += p_uv * np.ma.log2((p_uv / float(p_u * p_v)))
                        mutual_information = np.nan_to_num(mutual_information)
                    except:
                        i = 0
            probability_distribution[count1][count2] = np.zeros((2, 2))
            for first_case in range(np.shape(each_case)[0]):
                for second_case in range(np.shape(each_case)[0]):
                    probability_distribution[count1][count2][first_case][second_case] = each_case[first_case][
                                                                                            second_case] / float(
                        denominator)
    return probability_distribution_each_parameter, mutual_information, probability_distribution


def get_probability(probability_distribution, parameter_1, parameter_2, value_1, value_2):
    try:
        probability = probability_distribution[parameter_1][parameter_2][value_1][value_2]
    except:
        try:
            probability = probability_distribution[parameter_2][parameter_1][value_2][value_1]
        except:
            return 0
    return probability


def update_joint_probability_distribution(joint_probability_distribution, probability_distribution_each_parameter,
                                          probability_distribution, train_dataset, k):
    joint_probability_distribution_new = dict()
    for each_feature in range(np.shape(train_dataset)[1]):
        joint_probability_distribution_new[each_feature] = dict()
        for each_feature_2 in range(np.shape(train_dataset)[1]):
            try:
                if joint_probability_distribution[k][each_feature][each_feature_2]:
                    joint_probability_distribution_new[each_feature][each_feature_2] = np.zeros((2, 2))
                    for each_parameter in range(2):
                        for each_parameter_2 in range(2):
                            joint_probability_distribution_new[each_feature][each_feature_2][each_parameter][
                                each_parameter_2] = get_probability(
                                probability_distribution, each_feature, each_feature_2, each_parameter,
                                each_parameter_2)
            except:
                continue
        try:
            joint_probability_distribution_new[each_feature][-1] = probability_distribution_each_parameter[each_feature]
        except:
            i = 1
    joint_probability_distribution[k] = joint_probability_distribution_new
    return joint_probability_distribution


def e_step(train_dataset, k, joint_probability_distribution, mixture_probabilities, k_trees):
    num_of_examples = np.shape(train_dataset)[0]
    num_of_features = np.shape(train_dataset)[1]
    completed_data = np.zeros((num_of_examples, k))
    for each_example in range(num_of_examples):
        for each_k in range(k):
            completed_data[each_example][each_k] = mixture_probabilities[each_k]
            for each_feature in range(num_of_features):
                for each_feature_2 in range(each_feature, num_of_features):
                    try:
                        if k_trees[each_k][each_feature][each_feature_2] != 0:
                            if train_dataset[each_example][each_feature] == 0 and train_dataset[each_example][
                                each_feature_2] == 0:
                                completed_data[each_example][each_k] = (completed_data[each_example][each_k] *
                                                                        joint_probability_distribution[each_k][
                                                                            each_feature][
                                                                            each_feature_2][0][0]) / float(
                                    joint_probability_distribution[each_k][each_feature][-1])
                            if train_dataset[each_example][each_feature] == 0 and train_dataset[each_example][
                                each_feature_2] == 1:
                                completed_data[each_example][each_k] = (completed_data[each_example][each_k] *
                                                                        joint_probability_distribution[each_k][
                                                                            each_feature][
                                                                            each_feature_2][0][1]) / float(
                                    joint_probability_distribution[each_k][each_feature][-1])
                            if train_dataset[each_example][each_feature] == 1 and train_dataset[each_example][
                                each_feature_2] == 0:
                                completed_data[each_example][each_k] = (completed_data[each_example][each_k] *
                                                                        joint_probability_distribution[each_k][
                                                                            each_feature][
                                                                            each_feature_2][1][0]) / float(
                                    joint_probability_distribution[each_k][each_feature][-1])
                            if train_dataset[each_example][each_feature] == 1 and train_dataset[each_example][
                                each_feature_2] == 1:
                                completed_data[each_example][each_k] = (completed_data[each_example][each_k] *
                                                                        joint_probability_distribution[each_k][
                                                                            each_feature][
                                                                            each_feature_2][1][1]) / float(
                                    joint_probability_distribution[each_k][each_feature][-1])
                    except:
                        i = 1
        denominator_for_normalization = np.sum(completed_data[each_example])
        for each_k in range(k):
            completed_data[each_example][each_k] = completed_data[each_example][each_k] / float(
                denominator_for_normalization)
    return completed_data


def m_step(train_dataset, k, completed_data, joint_probability_distribution, r):
    num_of_examples = np.shape(train_dataset)[0]
    updated_mixture_probabilities = np.zeros((k, 1))
    for each_k in range(k):
        updated_mixture_probabilities[each_k] = np.sum(completed_data[:, each_k]) / float(num_of_examples)
    k_trees = dict()
    updated_joint_probability_distribution = dict()
    for each_k in range(k):
        probability_distribution_each_parameter, mutual_information, probability_distribution = compute_mutual_information(
            train_dataset, completed_data, each_k)
        zero_mutual_information_indices_feature_1 = np.reshape(np.random.choice(np.shape(mutual_information)[0], r),
                                                               (r, 1))
        zero_mutual_information_indices_feature_2 = np.reshape(np.random.choice(np.shape(mutual_information)[0], r),
                                                               (r, 1))
        zero_indices = np.concatenate(
            (zero_mutual_information_indices_feature_1, zero_mutual_information_indices_feature_2), axis=1)
        for each_row in zero_indices:
            mutual_information[each_row[0], each_row[1]] = 0
        spanning_tree = tree_bayesian_network.find_max_spanning_tree(mutual_information)
        k_trees[each_k] = spanning_tree
        updated_joint_probability_distribution = update_joint_probability_distribution(joint_probability_distribution,
                                                                                       probability_distribution_each_parameter,
                                                                                       probability_distribution,
                                                                                       train_dataset, each_k)
    return updated_mixture_probabilities, updated_joint_probability_distribution, k_trees


def test_log_likelihood(test_dataset, mixture_probabilities, joint_probability_distribution, k_trees, k):
    num_of_examples = np.shape(test_dataset)[0]
    num_of_features = np.shape(test_dataset)[1]
    log_likelihood = 0
    for each_example in test_dataset:
        log_probability_of_example = 0
        for each_k in range(k):
            log_probability_of_example_for_each_k = 0
            edges_dict, edges = tree_bayesian_network.get_edges(k_trees[each_k])
            for each_edge_set in edges:
                if each_example[each_edge_set[0]] == 0 and each_example[
                    each_edge_set[1]] == 0:
                    try:
                        log_probability_of_example_for_each_k += log2(
                            joint_probability_distribution[each_k][each_edge_set[0]][each_edge_set[1]][0][0]) - log2(
                            joint_probability_distribution[each_k][each_edge_set[1]][-1])
                    except:
                        i = 0
                if each_example[each_edge_set[0]] == 1 and each_example[
                    each_edge_set[1]] == 0:
                    try:
                        log_probability_of_example_for_each_k += log2(
                            joint_probability_distribution[each_k][each_edge_set[0]][each_edge_set[1]][1][0]) - log2(
                            joint_probability_distribution[each_k][each_edge_set[1]][-1])
                    except:
                        i = 0
                if each_example[each_edge_set[0]] == 0 and each_example[
                    each_edge_set[1]] == 1:
                    try:
                        log_probability_of_example_for_each_k += log2(
                            joint_probability_distribution[each_k][each_edge_set[0]][each_edge_set[1]][0][1]) - log2(
                            joint_probability_distribution[each_k][each_edge_set[1]][-1])
                    except:
                        i = 0
                if each_example[each_edge_set[0]] == 1 and each_example[
                    each_edge_set[1]] == 1:
                    try:
                        log_probability_of_example_for_each_k += log2(
                            joint_probability_distribution[each_k][each_edge_set[0]][each_edge_set[1]][1][1]) - log2(
                            joint_probability_distribution[each_k][each_edge_set[1]][-1])
                    except:
                        i = 0
            log_probability_of_example += log2(mixture_probabilities[each_k]) + log_probability_of_example_for_each_k
        log_likelihood = log_likelihood + log_probability_of_example
    return log_likelihood


def creating_k_bags(train_dataset, k):
    num_of_examples = np.shape(train_dataset)[0]
    bags = dict()
    for each_k in range(k):
        data_in_this_bag = np.random.choice(num_of_examples, num_of_examples, replace=True)
        bags[each_k] = train_dataset[data_in_this_bag]
    return bags


def run_model(train_dataset, test_dataset, valid_dataset, k, r, num_of_iterations, num_of_iterations_for_em):
    log_likelihood_for_each_iteration = np.zeros((num_of_iterations, 1))
    for each_iteration in range(num_of_iterations):
        train_dataset_bags = creating_k_bags(train_dataset, k)
        mixture_probabilities = initialize_mixture_probabilities(k)
        for each_k in range(k):
            test_log_likelihood = run_model_em(train_dataset_bags[each_k], test_dataset, valid_dataset, k, r,
                                               num_of_iterations_for_em)

            log_likelihood_for_each_iteration[each_iteration] = log_likelihood_for_each_iteration[
                                                                    each_iteration] + test_log_likelihood + np.ma.log2(
                mixture_probabilities[each_k])
        log_likelihood_for_each_iteration[each_iteration] = (log_likelihood_for_each_iteration[each_iteration]) / float(
            k)
    log_likelihood_mean = np.mean(log_likelihood_for_each_iteration)
    log_likelihood_standard_deviation = np.std(log_likelihood_for_each_iteration)
    return log_likelihood_mean, log_likelihood_standard_deviation


def run_model_em(train_dataset, test_dataset, valid_dataset, k, r, num_of_iterations_for_em):
    k_trees = initialize_k_trees(train_dataset, k)
    mixture_probabilities = initialize_mixture_probabilities(k)
    joint_probability_distribution = initialize_joint_probability_distribution(k, train_dataset, k_trees)
    converged = False
    for each_iteration_for_em in range(num_of_iterations_for_em):
        if converged:
            break;
        else:
            completed_data = e_step(train_dataset, k, joint_probability_distribution, mixture_probabilities,
                                    k_trees)
            updated_mixture_probabilities, updated_joint_probability_distribution, k_trees = m_step(train_dataset,
                                                                                                    k,
                                                                                                    completed_data,
                                                                                                    joint_probability_distribution,
                                                                                                    r)
            if not converged:
                for each_k in range(k):
                    if np.allclose(updated_mixture_probabilities, mixture_probabilities, 1 / 10, 1 / 10):
                        converged = not converged
            mixture_probabilities = updated_mixture_probabilities
            joint_probability_distribution = updated_joint_probability_distribution
    log_likelihood_for_this_iteration = test_log_likelihood(test_dataset, mixture_probabilities,
                                                            joint_probability_distribution, k_trees,
                                                            k)
    return log_likelihood_for_this_iteration


def validation_of_model(dataset_name, num_of_iterations):
    train_dataset, test_dataset, valid_dataset = import_data.import_data(dataset_name)
    k = range(5, 21, 5)
    r = range(10, 100, 10)
    best_k = 5
    best_r = 10
    best_log_likelihood = -math.inf
    for each in itertools.product(k, r):
        # Here I am testing on the validation dataset
        mean, standard_deviation = run_model(train_dataset, valid_dataset, test_dataset, each[0], each[1], 10, 100)
        if mean > best_log_likelihood:
            best_k = each[0]
            best_r = each[1]
    log_likelihood_mean_final, log_likelihood_standard_deviation_final = run_model(train_dataset, test_dataset,
                                                                                   valid_dataset, best_k, best_r,
                                                                                   10, 100)
    return log_likelihood_mean_final, log_likelihood_standard_deviation_final


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
log_likelihood_mean, log_likelihood_standard_deviation = validation_of_model(
                data_set_name, number_of_iterations)
print("The log likelihood mean and standard deviation are ", str(log_likelihood_mean),
                  str(log_likelihood_standard_deviation))print("The log likelihood mean and standard deviation are ", str(mean), str(standard_deviation))

If the above code does not run please use the following code 

log_likelihood_mean, log_likelihood_standard_deviation = mixture_of_trees_using_random_forest.run_model(data_set_name, 10, 50, 10, 100)
print(str(log_likelihood_mean), str(log_likelihood_standard_deviation))

"""