import rbf
import numpy as np
import matplotlib.pyplot as plt
from project_dataset_iterators import ValidationKFold


iris = np.loadtxt('irisData.txt', delimiter=',')

#Normalize by subracting mean and diving by standard dev
iris[:, :4] = iris[:, :4]-iris[:, :4].mean(axis=0)
iris[:, :4] = iris[:, :4]/iris[:, 4].std()

y = iris[:, 4:5]
y.shape = (y.shape[0], )

#Create target vector based on classes (three classes, one binary element for each)
targets = np.zeros((np.shape(iris)[0], 3))
indices = np.where(iris[:, 4] == 0)
targets[indices, 0] = 1
indices = np.where(iris[:, 4] == 1)
targets[indices, 1] = 1
indices = np.where(iris[:, 4] == 2)
targets[indices, 2] = 1

#random shuffle
"""
order = range(y.shape[0])
np.random.shuffle(order)
iris[:, :4] = iris[order, :4]
targets[:, :] = targets[order, :]
y = y[order]
"""

#K-Fold cross-validation sets and training
nFolds = 10
kf = ValidationKFold(iris.shape[0], nFolds, shuffle=True)


#Train models and get lists of errors for each fold
rbf_train_score = []
rbf_valid_score = []
test_scores = []
test_indexed_results = []
for train_indices, valid_indices, test_indices in kf:

    train = iris[train_indices]
    train_tgt = targets[train_indices]

    valid = iris[valid_indices]
    valid_tgt = targets[valid_indices]

    test = iris[test_indices]
    test_tgt = targets[test_indices]

    """RBF"""
    rbfnet = rbf.RBF(train, train_tgt, 10, .5)
    rbfnet.rbftrain(train, train_tgt, 0.00001, 1000, valid, valid_tgt)

    #Get training and validation scores on this Fold
    rbf_train_score.append(rbfnet.train_error)
    rbf_valid_score.append(rbfnet.valid_error)

    test_results = rbfnet.rbf_score(test, test_tgt)
    test_scores.append(test_results)


    rbf_train_score = np.array(rbf_train_score)
    rbf_valid_score = np.array(rbf_valid_score)

    mean_rbf_train_score = np.mean(rbf_train_score, axis=0)
    mean_rbf_valid_score = np.mean(rbf_valid_score, axis=0)

    plt.figure()
    plt.title("RBF Training and Validation")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Score")

    x = range(rbfnet.train_error.__len__())

    plt.plot(x, mean_rbf_train_score, "g", label="Training Score")
    plt.plot(x, mean_rbf_valid_score, "r", label="Validation Score")

    #plt.show()

    print test_scores
    print np.sum(test_scores)
    scores.append(np.sum(test_scores))



x = range(0, scores.__len__())

plt.plot(x, scores)
plt.show