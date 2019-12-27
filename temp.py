import import_data
import independent_bayesian_network
import numpy as np


train_dataset, test_dataset, valid_dataset = import_data.import_data("nltcs")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("nltcs " + str(log_likelihood))

train_dataset, test_dataset, valid_dataset = import_data.import_data("kdd")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("kdd " + str(log_likelihood))

train_dataset, test_dataset, valid_dataset = import_data.import_data("r52")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("r52 " + str(log_likelihood))

train_dataset, test_dataset, valid_dataset = import_data.import_data("msnbc")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("msnbc " + str(log_likelihood))

train_dataset, test_dataset, valid_dataset = import_data.import_data("baudio")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("baudio " + str(log_likelihood))

train_dataset, test_dataset, valid_dataset = import_data.import_data("bnetflix")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("bnetflix " + str(log_likelihood))

train_dataset, test_dataset, valid_dataset = import_data.import_data("accidents")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("accidents " + str(log_likelihood))

train_dataset, test_dataset, valid_dataset = import_data.import_data("plants")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("plants " + str(log_likelihood))


train_dataset, test_dataset, valid_dataset = import_data.import_data("jester")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("jester " + str(log_likelihood))

train_dataset, test_dataset, valid_dataset = import_data.import_data("dna")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("dna " + str(log_likelihood))

train_dataset, test_dataset, valid_dataset = import_data.import_data("nltcs")
parameters = independent_bayesian_network.fit(train_dataset)
log_likelihood = independent_bayesian_network.predict(test_dataset, parameters)
print("nltcs " + str(log_likelihood))

