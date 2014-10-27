import pcn
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


#K-Fold cross-validation sets and training
nFolds = 10
kf = ValidationKFold(iris.shape[0], nFolds, shuffle=True)

#Train models and get lists of errors for each fold
pcn_train_score = []
pcn_valid_score = []
test_scores = []
for train_indices, valid_indices, test_indices in kf:

    train = iris[train_indices]
    train_tgt = targets[train_indices]

    valid = iris[valid_indices]
    valid_tgt = targets[valid_indices]

    test = iris[test_indices]
    test_tgt = targets[test_indices]

    perceptron = pcn.pcn(train, train_tgt)
    train_scores, valid_scores = perceptron.pcntrainValid(train, train_tgt, valid, valid_tgt, 0.001, 1000)
    pcn_train_score.append(train_scores)
    pcn_valid_score.append(valid_scores)

    test = np.concatenate((test, -np.ones((test.shape[0], 1))), axis=1)
    print "FOLD"
    test_scores.append(perceptron.pcn_score(test, test_tgt))

print test_scores
print np.sum(test_scores)

pcn_train_score = np.array(pcn_train_score)
pcn_valid_score = np.array(pcn_valid_score)


mean_pcn_train_score = np.mean(pcn_train_score, axis=0)
mean_pcn_valid_score = np.mean(pcn_valid_score, axis=0)


plt.figure()
plt.title("PCN Training and Validation")
plt.xlabel("Number of Iterations")
plt.ylabel("Score")

x = range(pcn_valid_score.shape[1])

plt.plot(x, mean_pcn_train_score, "g", label="Training Score")
plt.plot(x, mean_pcn_valid_score, "r", label="Validation Score")



plt.show()

