
# Code from Chapter 12 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Code to run the decision tree on the Iris dataset
import dtree
import numpy as np
from project_dataset_iterators import ValidationKFold

# load Iris data
iris = np.loadtxt('irisData.txt',delimiter=',')

#format Iris data for decision tree
# decision tree need features, classes, and data corolating them

classes = iris[::,4::] #clasees are the 
iris = iris[::,:-1:]

# number of divions per variable
# number of features will be (number of variables) * (number of divisions)
divisions = 20 
features = ""

columnMins = iris.min(axis=0)
columnMaxs = iris.max(axis=0)
columsDiffs = []

# to set divisions must have max and min of each variable
for i in range(len(columnMins)):
    columsDiffs.append(columnMaxs[i] - columnMins[i]) 

for i in range(np.shape(iris)[1]):
    for j in range(divisions-1):
        features = features + "X"+str(i)+ ">"+ str(columnMins[i]+columsDiffs[i]*(j+1)/divisions) + ','
features  = features + 'FillerItem\n'
irisTreeData = ""

# make data formated for the tree
for i in range(np.shape(iris)[0]): # Go through the rows
    for j in range(len(columnMins)): # go through the coloums
        for z in range(divisions-1): #go through buckets for each datapoint
            if (iris[i][j]>columnMins[j]+columsDiffs[j]*(z+1)/divisions):
                irisTreeData = irisTreeData  + '1,'
            else:
                irisTreeData = irisTreeData  + '0,'

    # append the correct class to the vector
    tempclass = ""
    if (classes[i][0] == 0):
        tempclass = "Iris-setosa"
    if (classes[i][0] == 1):
        tempclass = "Iris-versicolor"
    if (classes[i][0] == 2):
        tempclass = "Iris-virginica"
    
    irisTreeData = irisTreeData + tempclass
    irisTreeData = irisTreeData  + '\n'
  
newInfo  = features + irisTreeData

f = open('formatted_iris_data.txt', 'w')
f.write(newInfo)
f.close()

#The data is now formated for the decision tree

#K-Fold validation
nFolds = 20
percentCorrect = []

kf = ValidationKFold(150, nFolds, shuffle=True)

kf_all = []
for train, valid, test in kf:
    kf_all.append([train, test, valid])
    
#kf_all = []

for i in range(nFolds): # lop through each fold case
    
    f = open('formatted_iris_data.txt', 'r')
    fTest = open('Test_data.txt','w')
    fTrain = open('Train_data.txt','w')
    fValid = open('Validation_data.txt','w')
    
    #write the data from the entire document to each specific case [train, test, and validate]
    features = f.readline()
    
    fTest.write(features)
    fTrain.write(features)
    fValid.write(features)
    
    
    for k in range(150):
        if k in kf_all[i][0]:  # check if the item is in the traning set
            fTrain.write(f.readline())
        elif k in kf_all[i][1]: # check if the item is in the test set
            fTest.write(f.readline())
        elif k in kf_all[i][2]: # check if the item is in the valid  set
            fValid.write(f.readline())

    
    fTest.close()
    fTrain.close()
    fValid.close()

    
    tree = dtree.dtree()
    data,classes,features = tree.read_data('Train_data.txt')
    Trained_Tree=tree.make_tree(data,classes,features)
 #   tree.printTree(Trained_Tree, ' ')
    Test_data,Test_classes,Test_features = tree.read_data('Test_data.txt')
    Test_guesses = tree.classifyAll(Trained_Tree,Test_data)
    
    
#    print "Test_Classes\n",Test_classes
#    print "Test_Guesses\n",Test_guesses


    
    correct = 0
    
    for x in range(len(Test_classes)):
        if Test_classes[x] == Test_guesses[x]:
            correct+=1
    
#    print "Error: ", float(correct)/len(Test_classes)
    percentCorrect.append(float(correct)/len(Test_classes))

# this is the avrage % correct over N folds.
print sum(percentCorrect) / len(percentCorrect)

# Varaibles we can change are number of divisions of the data into features and number of folds in the analysis