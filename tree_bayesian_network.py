from operator import itemgetter
import numpy as np
from numpy.ma import negative
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._min_spanning_tree import minimum_spanning_tree as mst

import import_data


def find_parameters(train_dataset):
    parameters = []
    for each_parameter in range(np.shape(train_dataset)[1]):
        value = (np.sum(train_dataset[:, each_parameter]) + 1) / (float(2 + np.shape(train_dataset)[0]))
        parameters.append(value)
    return parameters


def compute_mutual_information(train_dataset, parameters):
    num_of_examples = np.shape(train_dataset)[0]
    num_of_features = np.shape(train_dataset)[1]
    mutual_information = np.zeros((num_of_features, num_of_features))
    for count1 in range(num_of_features):
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
                        p_u = 1 - parameters[count1]
                    else:
                        p_u = parameters[count1]
                    if each_column == 0:
                        p_v = 1 - parameters[count2]
                    else:
                        p_v = parameters[count2]
                    mutual_information[count1][count2] += p_uv * np.ma.log2((p_uv / float(p_u * p_v)))
    return mutual_information


def find_max_spanning_tree(mutual_information):
    negative_mutual_inf = csr_matrix(negative(mutual_information))
    tree = mst(negative_mutual_inf)
    return negative(tree.toarray())


def get_edges(mst):
    non_zero_values = mst.nonzero()
    edges = np.vstack(non_zero_values).T
    edges.sort(axis=1)
    edge_dict = dict()
    for i in range(0, len(mst)):
        edge_dict[i] = set()
    for each_edge in edges:
        edge_dict[each_edge[0]].add(each_edge[1])
        edge_dict[each_edge[1]].add(each_edge[0])
    return edge_dict, edges


def test_log_likelihood(edge_dict, edges, test_data, parameters):
    log_likelihood = 0
    for each_edge_set in edges:
        for each_tuple in test_data:
            each_example = list(itemgetter(*each_edge_set)(each_tuple))
            if each_example[0] == 0:
                p_u = 1 - parameters[each_edge_set[0]]
            else:
                p_u = parameters[each_edge_set[0]]
            if each_example[1] == 0:
                p_v = 1 - parameters[each_edge_set[1]]
            else:
                p_v = parameters[each_edge_set[1]]
            log_likelihood = log_likelihood + np.ma.log2(p_u * p_v)
            log_likelihood = log_likelihood - ((edge_dict[each_edge_set[0]]).__len__() - 1) * np.ma.log2(p_v)
    return log_likelihood


def run_model(dataset_name):
    train_dataset, test_dataset, valid_dataset = import_data.import_data(dataset_name)
    parameters = find_parameters(train_dataset)
    mutual_information = compute_mutual_information(train_dataset, parameters)
    mst = find_max_spanning_tree(mutual_information)
    edges_dict, edges = get_edges(mst)
    test_log_likelihood_score = test_log_likelihood(edges_dict, edges, test_dataset, parameters)
    return test_log_likelihood_score


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
train_dataset, test_dataset, valid_dataset = import_data.import_data(data_set_name)
parameters = find_parameters(train_dataset)
mutual_information = compute_mutual_information(train_dataset, parameters)
mst = find_max_spanning_tree(mutual_information)
edges_dict, edges = get_edges(mst)
test_log_likelihood = test_log_likelihood(edges_dict, edges, test_dataset, parameters)
print("The log likelihood is " + str(test_log_likelihood))

"""