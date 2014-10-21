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
order = range(y.shape[0])
np.random.shuffle(order)
iris[:, :4] = iris[order, :4]
targets[:, :] = targets[order, :]
y = y[order]


#K-Fold cross-validation sets and training
nFolds = 10
kf = ValidationKFold(iris.shape[0], nFolds, shuffle=True)

for train_indices, valid_indices, test_indices in kf:


    train = iris[train_indices]
    train_tgt = targets[train_indices]

    valid = iris[valid_indices]
    valid_tgt = targets[valid_indices]

    test = iris[test_indices]
    test_tgt = targets[test_indices]


    """RBF"""
    rbfnet = rbf.RBF(train, train_tgt, 10, .5)
    rbfnet.rbftrain(train, train_tgt, 0.00001, 5000, valid, valid_tgt)

    plt.figure()
    plt.subplot(2, 1, 1)
    x = range(rbfnet.train_error.__len__())

    plt.plot(x, rbfnet.train_error)

    plt.subplot(2, 1, 2)
    x = range(rbfnet.valid_error.__len__())

    plt.plot(x, rbfnet.valid_error, "r")
    print rbfnet.valid_error
    plt.show()

