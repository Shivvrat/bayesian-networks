# Bayesian Networks



## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project
In this project, you will implement the following four algorithms and test their performance on 10 datasets available on the class web page.

1. ***Independent Bayesian networks.***
Assume the following structure: The Bayesian networks has no edges. Learn the parameters of the independent Bayesian network using the maximum likelihood approach. Use 1-Laplace smoothing to ensure that you don’t have any zero probabilities in the model.

2. ***Tree Bayesian networks.***
Use the Chow-Liu algorithm to learn the structure and parameters of the Bayesian network. Use 1-Laplace smoothing to ensure that you don’t have any zeros when computing the mutual information as well as zero probabilities in the model. See section 2 in [Meila and Jordan, 2001].

3. ***Mixtures of Tree Bayesian networks using EM.***
The model is defined as follows. 
We have one latent variable having k values and each mixture component is a Tree Bayesian network. Learn the structure and parameters of the model using the EM-algorithm (in the M-step each mixture component is learned using the Chow-Liu algorithm). Select k using the validation set and use 1 - Laplace smoothing. Run the EM algorithm until convergence or until 100 iterations whichever is earlier. See section 3 in [Meila and Jordan, 2001].

4. ***Mixtures of Tree Bayesian networks using Random Forests.***
The model is defined as above (see Item (3)). Learn the structure and parameters of the model using the following Random-Forests style approach. Given two hyper-parameters (k, r), generate k sets of Bootstrap samples and learn the i-th Tree Bayesian network 1 using the i-th set of the Bootstrap samples by randomly setting exactly r mutual information scores to 0 (as before use the Chow-Liu algorithm with r mutual information scores set to 0 to learn the structure and parameters of the Tree Bayesian network). Select k and r using the validation set and use 1-Laplace smoothing. You can either set pi = 1/k for all i or use any reasonable method (reasonable method is
extra credit). Describe your (reasonable) method precisely in your report. Does it improve over the baseline approach that uses pi = 1/k.

### Built With

* [Python 3.7](https://www.python.org/downloads/release/python-370/)


## Getting Started

Lets see how to run this program on a local machine.

### Prerequisites

You will need the following modules 
```
1 import sys
2 import warnings 
3 import itertools 
4 import math
5 from operator import itemgetter 
6 from random import random 
7 import import_data 
8 import numpy as np 
9 import networkx
10 from numpy.ma import log2 
11 import numpy as np 
12 import pandas as pd 
13 import itertools 
14 import math
15 from operator import itemgetter 
16 from random import random 
17 import import_data 
18 import numpy as np 
19 import networkx
20 from numpy.ma import log2
21 from operator import itemgetter 
22 import numpy as np
23 from numpy.ma import negative
24 from scipy.sparse import csr_matrix 
25 from scipy.sparse.csgraph._min_spanning_tree import minimum_spanning_tree as mst
```
### Installation

1. Clone the repo
```sh
https://github.com/Shivvrat/bayesian-networks.git
```
Use the main.py to run the algorithm.


<!-- USAGE EXAMPLES -->
## Usage
Please enter the following command line argument :-
```sh
python main.py <dataset_name > <algorithm_name > <parameter_1 >
```
Please use the following command line parameters for the main.py file :-

***Part 1.*** The code should look like
```python main.py <dataset -name > -ibn```

We only have 1 parameter for the part1 which is: 
```< dataset−name >```
We need to provide the dataset name for which we want to run the algorithm.

***Part 2.*** The code should look like :
```python main.py <dataset -name > -tbn```

We only have 1 parameter for the part 2 which is: 

```< dataset−name >``` 
We need to provide the dataset name for which we want to run the algorithm.
    
***Part 3.*** The code should look like :
```python main.py <dataset -name > -mtem <number -of -iterations >```

We have 2 parameters for the part 3 algorithm which is: 
```< dataset−name >```
We need to provide the dataset name for which we want to run the algorithm. 

```< number − of − iterations >```
We need to provide the number of iterations for which we want to run the algorithm.

***Part 4***  The code should look like :
```python main.py <dataset -name > -mtemrf <number -of -iterations >```

We have 2 parameters for the part 4 algorithm which is: 
```< dataset−name >```
We need to provide the dataset name for which we want to run the algorithm. 

```< number − of − iterations >```
We need to provide the number of iterations for which we want to run the algorithm.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - Shivvrat Arya[@ShivvratA](https://twitter.com/ShivvratA) - shivvratvarya@gmail.com

Project Link: [https://github.com/Shivvrat/bayesian-networks.git](https://github.com/Shivvrat/bayesian-networks.git)



