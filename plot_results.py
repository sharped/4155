import matplotlib.pyplot as plt
import numpy as np

svm_rbf = [143.0, 143, 143, 142, 144, 143, 143, 144, 144, 142]
svm_lin = [140.0, 135, 141, 143, 143, 141, 141, 140, 145, 140]

mlp = [144, 142, 148, 144, 144, 143, 144, 145, 142, 145]

rbf_net = [133, 129, 126, 122, 130, 126, 133, 135, 140, 134]

pcn = [135, 130, 135, 133.0, 133, 141, 135, 134, 135, 135, 134]

x = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5]


y = []
y.extend(svm_rbf)
y.extend(svm_lin)
y.extend(mlp)
y.extend(rbf_net)
y.extend(pcn)

print y.__len__()
print x.__len__()

plt.figure()
plt.title("Model Comparison")
plt.xlabel("Model")
plt.ylabel("Scores")

plt.scatter(x, y)

plt.show()

