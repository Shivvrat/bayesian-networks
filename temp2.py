import import_data
import tree_bayesian_network
import numpy as np


train_dataset, test_dataset, valid_dataset = import_data.import_data("nltcs")
parameters = tree_bayesian_network.find_parameters(train_dataset)
mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
edges_dict, edges  = tree_bayesian_network.get_edges(mst)
test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict,edges,test_dataset,parameters)
print("nltcs " + str(test_log_likelihood))

"""
train_dataset, test_dataset, valid_dataset = import_data.import_data("kdd")
parameters = tree_bayesian_network.find_parameters(train_dataset)
mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
edges_dict, edges  = tree_bayesian_network.get_edges(mst)
test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict,edges,test_dataset,parameters)
print "kdd " + str(test_log_likelihood)

train_dataset, test_dataset, valid_dataset = import_data.import_data("r52")
parameters = tree_bayesian_network.find_parameters(train_dataset)
mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
edges_dict, edges  = tree_bayesian_network.get_edges(mst)
test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict,edges,test_dataset,parameters)
print "r52 " + str(test_log_likelihood)

"""
"""
train_dataset, test_dataset, valid_dataset = import_data.import_data("msnbc")
parameters = tree_bayesian_network.find_parameters(train_dataset)
mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
edges_dict, edges  = tree_bayesian_network.get_edges(mst)
test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict,edges,test_dataset,parameters)
print "msnbc " + str(test_log_likelihood)

train_dataset, test_dataset, valid_dataset = import_data.import_data("baudio")
parameters = tree_bayesian_network.find_parameters(train_dataset)
mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
edges_dict, edges  = tree_bayesian_network.get_edges(mst)
test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict,edges,test_dataset,parameters)
print "baudio " + str(test_log_likelihood)

train_dataset, test_dataset, valid_dataset = import_data.import_data("bnetflix")
parameters = tree_bayesian_network.find_parameters(train_dataset)
mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
edges_dict, edges  = tree_bayesian_network.get_edges(mst)
test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict,edges,test_dataset,parameters)
print "bnetflix " + str(test_log_likelihood)

train_dataset, test_dataset, valid_dataset = import_data.import_data("accidents")
parameters = tree_bayesian_network.find_parameters(train_dataset)
mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
edges_dict, edges  = tree_bayesian_network.get_edges(mst)
test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict,edges,test_dataset,parameters)
print "accidents " + str(test_log_likelihood)

train_dataset, test_dataset, valid_dataset = import_data.import_data("plants")
parameters = tree_bayesian_network.find_parameters(train_dataset)
mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
edges_dict, edges  = tree_bayesian_network.get_edges(mst)
test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict,edges,test_dataset,parameters)
print "plants " + str(test_log_likelihood)

train_dataset, test_dataset, valid_dataset = import_data.import_data("dna")
parameters = tree_bayesian_network.find_parameters(train_dataset)
mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
edges_dict, edges  = tree_bayesian_network.get_edges(mst)
test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict,edges,test_dataset,parameters)
print("dna " + str(test_log_likelihood))

train_dataset, test_dataset, valid_dataset = import_data.import_data("jester")
parameters = tree_bayesian_network.find_parameters(train_dataset)
mutual_information = tree_bayesian_network.compute_mutual_information(train_dataset, parameters)
mst = tree_bayesian_network.find_max_spanning_tree(mutual_information)
edges_dict, edges  = tree_bayesian_network.get_edges(mst)
test_log_likelihood = tree_bayesian_network.test_log_likelihood(edges_dict,edges,test_dataset,parameters)
print "jester " + str(test_log_likelihood)


"""