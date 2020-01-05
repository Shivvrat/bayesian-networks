# Bayesian Networks
Inducing Decision Trees
In this homework you will implement and test the decision tree learning
algorithm (See Mitchell, Chapter 3). It is acceptable to look at Java code
for decision trees in WEKA. However, you cannot copy code from WEKA.
You can use either C/C++, Java or Python to implement your algorithms.
Your C/C++ implementations should compile on Linux gcc/g++
compilers.
 Download the 15 datasets available on the class web page. Each data
set is divided into three sub-sets: the training set, the validation set
and the test set. Data sets are in CSV format. Each line is a training
(or test) example that contains a list of attribute values separated by
a comma. The last attribute is the class-variable. Assume that all
attributes take values from the domain f0,1g.
The datasets are generated synthetically by randomly sampling solutions
and non-solutions (with solutions having class \1" and nonsolutions
having class \0") from a Boolean formula in conjunctive normal
form (CNF). I randomly generated ve formulas having 500 variables
and 300, 500, 1000, 1500 and 1800 clauses (where the length
of each clause equals 3) respectively and sampled 100, 1000 and 5000
positive and negative examples from each formula. I am using the
following naming convention for the les. Filenames train, test
and valid denote the training, test and validation data respectively.
train c[i] d[j]:csv where i and j are integers contains training data
