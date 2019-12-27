import import_data
import EM_using_random_forest
"""
log_likelihood_mean, log_likelihood_standard_deviation = mixture_of_trees_using_random_forest.run_model('nltcs', 10, 10,
                                                                                                        2)

print("nltcs ", str(log_likelihood_mean), str(log_likelihood_standard_deviation))
log_likelihood_mean, log_likelihood_standard_deviation = mixture_of_trees_using_random_forest.run_model('kdd', 10, 10,
                                                                                                        2)
print("kdd ", str(log_likelihood_mean), str(log_likelihood_standard_deviation))

train_dataset, test_dataset, valid_dataset = import_data.import_data("msnbc")
log_likelihood_mean, log_likelihood_standard_deviation = mixture_of_trees_using_random_forest.run_model(train_dataset, test_dataset, valid_dataset, 2, 10,
                                                                                                        1)

print("msnbc ", str(log_likelihood_mean), str(log_likelihood_standard_deviation))

train_dataset, test_dataset, valid_dataset = import_data.import_data("baudio")
log_likelihood_mean, log_likelihood_standard_deviation = mixture_of_trees_using_random_forest.run_model(train_dataset, test_dataset, valid_dataset, 2, 10,
                                                                                                        1)
print("baudio ", str(log_likelihood_mean), str(log_likelihood_standard_deviation))

train_dataset, test_dataset, valid_dataset = import_data.import_data("bnetflix")
log_likelihood_mean, log_likelihood_standard_deviation = mixture_of_trees_using_random_forest.run_model(train_dataset, test_dataset, valid_dataset, 2, 10,
                                                                                                        1)
print("bnetflix ", str(log_likelihood_mean), str(log_likelihood_standard_deviation))

train_dataset, test_dataset, valid_dataset = import_data.import_data("accidents")
log_likelihood_mean, log_likelihood_standard_deviation = mixture_of_trees_using_random_forest.run_model(train_dataset, test_dataset, valid_dataset, 2, 10,
                                                                                                        1)
print("accidents ", str(log_likelihood_mean), str(log_likelihood_standard_deviation))

train_dataset, test_dataset, valid_dataset = import_data.import_data("plants")
log_likelihood_mean, log_likelihood_standard_deviation = EM_using_random_forest.validation_of_model("plants", 1)
print("plants ", str(log_likelihood_mean), str(log_likelihood_standard_deviation))
"""
train_dataset, test_dataset, valid_dataset = import_data.import_data("dna")
log_likelihood_mean, log_likelihood_standard_deviation = EM_using_random_forest.run_model(train_dataset, test_dataset, valid_dataset, 2, 10,
                                                                                                        1, 1)
print("dna ", str(log_likelihood_mean), str(log_likelihood_standard_deviation))


train_dataset, test_dataset, valid_dataset = import_data.import_data("jester")
log_likelihood_mean, log_likelihood_standard_deviation = EM_using_random_forest.validation_of_model("jester", 1)
print("jester ", str(log_likelihood_mean), str(log_likelihood_standard_deviation))

