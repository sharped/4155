import mlp
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

#Train models and get lists of errors for each fold
mlp_train_score = []
mlp_valid_score = []
for train_indices, valid_indices, test_indices in kf:

    train = iris[train_indices]
    train_tgt = targets[train_indices]

    valid = iris[valid_indices]
    valid_tgt = targets[valid_indices]

    test = iris[test_indices]
    test_tgt = targets[test_indices]

    """MLP"""
    mlpnet = mlp.mlp(train, train_tgt, 20, beta=1, momentum=0.9, outtype='logistic')
    mlpnet.mlptrain(train, train_tgt, 0.01, 1000, valid, valid_tgt)

    #Get training and validation scores on this Fold
    mlp_train_score.append(mlpnet.train_scores_list)
    mlp_valid_score.append(mlpnet.valid_scores_list)


mlp_train_score = np.array(mlp_train_score)
mlp_valid_score = np.array(mlp_valid_score)


mean_mlp_train_score = np.mean(mlp_train_score, axis=0)
mean_mlp_valid_score = np.mean(mlp_valid_score, axis=0)


plt.figure()
plt.title("RBF Network Results")
plt.xlabel("Number of Iterations")
plt.ylabel("Score")

x = range(mlpnet.train_scores_list.__len__())

plt.plot(x, mean_mlp_train_score, "g", label="Training Score")
plt.plot(x, mean_mlp_valid_score, "r", label="Validation Score")

plt.show()


