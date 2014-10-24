
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import pcn as pcn
import kmeans

class RBF:
    """ A Multi-Layer Perceptron"""

    def __init__(self, inputs, targets, nRBF, sigma, outtype='logistic', normalize=False):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nRBF = nRBF
        self.sigma = sigma
        self.normalize = normalize

        self.outtype = outtype

        self.train_error = []
        self.valid_error = []

        self.weights1 = np.ones((self.nin, self.nRBF))
        self.hidden = np.ones((self.ndata, self.nRBF))

        self.pcn = pcn.pcn(self.hidden, targets)

    def earlystopping(self, inputs, targets, valid, validtargets, eta, niterations=100):

        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000

        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count += 1
            print count
            self.rbftrain(inputs, targets, eta, niterations, valid, validtargets)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.rbffwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)


        print "Stopped", new_val_error, old_val_error1, old_val_error2
        return new_val_error

    def rbftrain(self, inputs, targets, eta, niterations, validation_inputs, vtargets):
        """ Train the thing """

        n_valid_points = validation_inputs.shape[0]

        indices = range(self.ndata)
        np.random.shuffle(indices)

        #Set RBF centers at randomly chosen datapoints
        for i in range(self.nRBF):
            self.weights1[:, i] = inputs[indices[i], :]

        #Calculate activations for training and validation sets
        self.valid_hidden = np.ones((n_valid_points, self.nRBF))
        for i in range(self.nRBF):
            self.hidden[:, i] = np.exp(-np.sum((inputs - np.ones((1, self.nin))*self.weights1[:, i])**2, axis=1)/(2*self.sigma**2))
            self.valid_hidden[:, i] = np.exp(-np.sum((validation_inputs - np.ones((1, self.nin))*self.weights1[:, i])**2, axis=1)/(2*self.sigma**2))

        if self.normalize:
            self.hidden[:, :-1] /= np.transpose(np.ones((1, self.hidden.shape[0]))*self.hidden[:, :-1].sum(axis=1))

        self.train_error, self.valid_error = self.pcn.pcntrainValid(self.hidden[:, :], targets, self.valid_hidden, vtargets, eta, niterations)


    def rbffwd(self, inputs):
        """ Run the network forward """
        self.hidden = np.zeros((inputs.shape[0], self.nRBF))

        for i in range(self.nRBF):
            self.hidden[:, i] = np.exp(-np.sum((inputs - np.ones((1, inputs.shape[1]))*self.weights1[:, i])**2, axis=1)/(2*self.sigma**2))

        if self.normalize:
            self.hidden[:, :-1] /= np.transpose(np.ones((1, self.hidden.shape[0]))*self.hidden[:, :-1].sum(axis=1))

        pcninputs = np.concatenate((self.hidden, -np.ones((inputs.shape[0], 1))), axis=1)
        outputs = self.pcn.pcnfwd(pcninputs)

        return outputs

    def confmat(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        outputs = self.mlpfwd(inputs)

        nclasses = np.shape(targets)[1]

        if nclasses == 1:
            nclasses = 2
            outputs = np.where(outputs > 0.5, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0)*np.where(targets == j, 1, 0))

        print "Confusion matrix is:"
        print cm
        print "Percentage Correct: ",np.trace(cm)/np.sum(cm)*100
