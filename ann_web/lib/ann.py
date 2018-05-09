import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils import *
from NeuralNetwork import NeuralNetwork

class ANN:

    def __init__(self,X_train, X_test, y_train, y_test, dimension, classes, filename):
        self.filename = filename
        self.X_train =  X_train
        self.X_test = X_test
        self.y_train =  y_train
        self.y_test = y_test
        self.dimension = dimension
        self.classes = classes

    def get_model(self):
        return pickle.load(open(self.filename, 'rb'))

    def write_model(self, trained_model):
        pickle.dump(trained_model, open(self.filename, 'wb'))

    def train(self):
        start = time.time()
        # Initialize Neural Network-> This setup has been tested to give the most optimal results. 
        ann = NeuralNetwork(n_output=self.dimension, 
                            n_features=self.X_train.shape[1], 
                            n_hidden=100, 
                            learning_rate=0.001,
                            epochs=300, 
                            momentum_const=0.5, 
                            decay_rate=0.00001, 
                            activation='sigmoid',
                            dropout=True,
                            minibatch_size=50, 
                            nesterov=True,
                            check_gradients=False)
        if(os.path.exists(self.filename)):
            print("Model already exists, will now calculate accuracy")
            ann = self.get_model()
        else:
            print("Model not found, we will proceed to train and calculate accuracy")
            ann.fit(self.X_train, self.y_train)
            self.write_model(ann)

        print ("The program exectuted successfuly in: %s seconds" % (time.time() - start))
        return ann
          
        
    def accuracy(self, classifier):
        accuracy = calculate_accuracy(classifier, self.X_test, self.y_test)
        train_error, test_error = get_train_test_error(classifier, self.X_test, self.y_test, num_iterations=1, split= 0.33)
        # print ("Training Error: {}, Testing error: {} ".format(train_error, test_error))
        # print ("Total accuracy {}".format(accuracy))

        #Building a confusion Matrix
        # pred = classifier.predict(x_test)
        # cm = confusion_matrix(y_test, pred)
        # np.set_printoptions(precision=2)
        # plot_confusion_matrix(cm, classes)
        return accuracy, train_error, test_error
   
