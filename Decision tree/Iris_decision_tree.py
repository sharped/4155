
# Code from Chapter 12 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Code to run the decision tree on the Party dataset
import dtree
import numpy as np
import project_dataset_iterators as ValidationKFold

'''
            Format iris data
'''



iris = np.loadtxt('irisData.txt',delimiter=',')


nFolds = 10

kf = ValidationKFold(iris.shape[0], nFolds, shuffle=True)

#SVM does not use validation set so throw it out
kf_noValid = []
for train, valid, test in kf:
    kf_noValid.append([train, test])
    
print noValid




classes = iris[::,4::]
iris = iris[::,:-1:]


divisions = 20 # number of divions per data point
features = ""

columnMins = iris.min(axis=0)
columnMaxs = iris.max(axis=0)
columsDiffs = []
for i in range(len(columnMins)):
    columsDiffs.append(columnMaxs[i] - columnMins[i]) 
    
for i in range(np.shape(iris)[1]):
    for j in range(divisions-1):
        features = features + "X"+str(i)+ ">"+ str(columnMins[i]+columsDiffs[i]*(j+1)/divisions) + ','
features  = features + 'FillerItem\n'

# features = 'A1,A2,A3,A4,B1,B2,B3,B4,C1,C2,C3,C4,D1,D2,D3,D4,FillerItem\n <- Example of what shoudl be going on above

   

#irisTreeData = np.zeros((np.shape(iris)[0],len(columnMins)*numberOfIntervals))
irisTreeData = ""
'''
New code for univariae trees
'''
for i in range(np.shape(iris)[0]): # Go through the rows
    for j in range(len(columnMins)): # go through the coloums
        for z in range(divisions-1): #go through buckets for each datapoint
            if (iris[i][j]>columnMins[j]+columsDiffs[j]*(z+1)/divisions):
                irisTreeData = irisTreeData  + 'Larger,'
            else:
                irisTreeData = irisTreeData  + 'Smaller,'
#           if (iris[i][j] == columnMins[j] + columsDiffs[j]):
#                irisTreeData = irisTreeData[:-2]  + '1,'

    tempclass = ""
    if (classes[i][0] == 0):
        tempclass = "Iris-setosa"
    if (classes[i][0] == 1):
        tempclass = "Iris-versicolor"
    if (classes[i][0] == 2):
        tempclass = "Iris-virginica"
    
    irisTreeData = irisTreeData + tempclass
    irisTreeData = irisTreeData  + '\n'
  
  
info  = features + irisTreeData


f = open('formatted_iris_data.txt', 'w')
f.write(info)
f.close()

tree = dtree.dtree()
data,classes,features = tree.read_data('formatted_iris_data.txt')
t=tree.make_tree(data,classes,features)
tree.printTree(t,' ')
'''
#iris = np.loadtxt('irisData.txt',delimiter=',')
#print iris


This is the old coed for a multivariate Tree
for i in range(np.shape(iris)[0]): # Go through the rows
    for j in range(len(columnMins)): # go through the coloums
        for z in range(buckets): #go through buckets for each datapoint
            if ((iris[i][j]>=columnMins[j]+columsDiffs[j]*z/buckets and iris[i][j]<columnMins[j]+columsDiffs[j]*(z+1)/buckets) or ((z+1)==buckets and iris[i][j]==columnMins[j]+columsDiffs[j])):
                irisTreeData = irisTreeData  + '1,'
            else:
                irisTreeData = irisTreeData  + '0,'
#           if (iris[i][j] == columnMins[j] + columsDiffs[j]):
#                irisTreeData = irisTreeData[:-2]  + '1,'

'''
