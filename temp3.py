import import_data
import mixture_of_trees_using_EM

"""
train_dataset, test_dataset, valid_dataset = import_data.import_data("nltcs")
mean, standard_deviation = mixture_of_trees_using_EM.run_model(train_dataset, test_dataset, valid_dataset, 2, 2, 2)
print(mean, standard_deviation)


train_dataset, test_dataset, valid_dataset = import_data.import_data("kdd")
mean, standard_deviation = mixture_of_trees_using_EM.run_model(train_dataset, test_dataset, valid_dataset, 2, 2, 2)
print("kdd ", mean, standard_deviation)

train_dataset, test_dataset, valid_dataset = import_data.import_data("msnbc")
mean, standard_deviation = mixture_of_trees_using_EM.run_model(train_dataset, test_dataset, valid_dataset, 2, 1, 2)
print("msnbc ", mean, standard_deviation)

train_dataset, test_dataset, valid_dataset = import_data.import_data("baudio")
mean, standard_deviation = mixture_of_trees_using_EM.run_model(train_dataset, test_dataset, valid_dataset, 2, 1, 2)
print("baudio ", mean, standard_deviation)

train_dataset, test_dataset, valid_dataset = import_data.import_data("bnetflix")
mean, standard_deviation = mixture_of_trees_using_EM.run_model(train_dataset, test_dataset, valid_dataset, 2, 1, 2)
print("bnetflix ",mean, standard_deviation)

train_dataset, test_dataset, valid_dataset = import_data.import_data("accidents")
mean, standard_deviation = mixture_of_trees_using_EM.run_model(train_dataset, test_dataset, valid_dataset, 2, 1, 2)
print("accidents ", mean, standard_deviation)

train_dataset, test_dataset, valid_dataset = import_data.import_data("plants")
mean, standard_deviation = mixture_of_trees_using_EM.run_model(train_dataset, test_dataset, valid_dataset, 2, 1, 2)
print("plants ", mean, standard_deviation)

train_dataset, test_dataset, valid_dataset = import_data.import_data("dna")
mean, standard_deviation = mixture_of_trees_using_EM.run_model(train_dataset, test_dataset, valid_dataset, 2, 1, 2)
print("dna ", mean, standard_deviation)
"""
train_dataset, test_dataset, valid_dataset = import_data.import_data("jester")
mean, standard_deviation = mixture_of_trees_using_EM.validation_of_model("jester", 1)
print("jester ", mean, standard_deviation)
