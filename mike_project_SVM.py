import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from project_dataset_iterators import ValidationKFold
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve

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

"""SVMs"""
#Create SVMs
svm_lin = LinearSVC(C=10)
svm_rbf = SVC(C=10, kernel="rbf", gamma=0.001)

#K-Fold cross-validation sets and training (Leave 1 out)
nFolds = 10
kf = ValidationKFold(iris.shape[0], nFolds, shuffle=True)

#SVM does not use validation set so throw it out
kf_noValid = []
for train, valid, test in kf:
    kf_noValid.append([train, test])

"""Train & Get Learning Curves"""
#RBF
train_sizes, train_scores, test_scores = learning_curve(svm_rbf, iris[:, :4], y, cv=kf_noValid, train_sizes=np.linspace(.1, 1.0, 20))

#Calc means
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

#Save the final results for later plotting
rbf_test_scores = test_scores[19:20, :]

#Plotting
plt.figure()
plt.subplot(2, 1, 1)
plt.title("SVM_RBF")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation score")

#Linear
train_sizes, train_scores, test_scores = learning_curve(svm_lin, iris[:, :4], y, cv=kf_noValid, train_sizes=np.linspace(.1, 1.0, 20))

#Calc Means
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

#Save the final results for later plotting
linear_test_scores = test_scores[19:20]

#Plotting
plt.subplot(2, 1, 2)
plt.title("SVM_Linear")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation score")

plt.show()

#Plot post-training-test scores
"""
NOT WORKING NOW
rbf_test_scores = []
linear_test_scores = []
for train, test in kf_noValid:
    print svm_rbf.score(iris[test], y[test])
    rbf_test_scores.append(svm_rbf.score(iris[test, :], y[test]))
    linear_test_scores.append(svm_lin.score(iris[test, :, y[test]]))

"""

plt.figure()
plt.title("Comparison of post-training test Results")
plt.xlabel("Model")
plt.ylabel("Score")

rbf_X_indices = np.ones((nFolds))
linear_X_indices = np.ones((nFolds))*2

x_indices = np.append(rbf_X_indices, linear_X_indices)
scores = np.append(rbf_test_scores, linear_test_scores)

plt.scatter(x_indices, scores)

plt.show()

