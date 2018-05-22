import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils import *
from NeuralNetwork import NeuralNetwork

class ANN:

    def __init__(self,filename, epochs, learning, decay_rate, hidden, new_model = 'on'):
        self.X_train, self.X_test, self.y_train, self.y_test, self.classes = prepare_data() 
        self.filename = filename
        self.epochs =  epochs
        self.new_model = new_model
        self.learning = learning
        self.hidden = hidden
        self.decay_rate =decay_rate


    def get_model(self):
        return pickle.load(open(self.filename, 'rb'))

    def write_model(self, trained_model):
        pickle.dump(trained_model, open(self.filename, 'wb'))

    def train(self):
        #start = time.time()
        # Initialize Neural Network-> This setup has been tested to give the most optimal results. 
        ann = NeuralNetwork(n_output=number_of_labels(self.classes), 
                            n_features=self.X_train.shape[1], 
                            n_hidden= self.hidden, 
                            learning_rate= self.learning,
                            epochs= self.epochs, 
                            momentum_const=0.5, 
                            decay_rate=0.00001, # self.decay_rate, 
                            activation='sigmoid',
                            dropout=True,
                            minibatch_size=50, 
                            nesterov=True,
                            check_gradients=False)
                            
        if (self.new_model=='on'):
             # Train a new model instead
            if(os.path.exists(self.filename)):
                os.remove(self.filename)
            ann.fit(self.X_train,self.y_train)
            self.write_model(ann)
        else:
           ann = self.get_model()

        # print ("The program exectuted successfuly in: %s seconds" % (time.time() - start))
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
    
   
