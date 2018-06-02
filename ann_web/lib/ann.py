import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import *
from NeuralNetwork import NeuralNetwork

class ANN:

    def __init__(self,filename, epochs, learning, decay_rate, hidden,momentum, 
            activation, new_model = 'on'):
            
        self.filename = filename
        self.X_train, self.X_test, self.y_train, self.y_test, self.classes = prepare_data(self.filename) 
        self.epochs =  epochs
        self.new_model = new_model
        self.learning = learning
        self.hidden = hidden
        self.decay_rate =decay_rate
        self.momentum = momentum
        self.activation = activation


    def get_model(self):
        return pickle.load(open(MODELFILENAME, 'rb'))

    def write_model(self, trained_model):
        pickle.dump(trained_model, open(MODELFILENAME, 'wb'))

    def train(self):
        #start = time.time()
        # Initialize Neural Network-> This setup has been tested to give the most optimal results. 
        ann = NeuralNetwork(n_output=number_of_labels(self.classes), 
                            n_features=self.X_train.shape[1], 
                            n_hidden= self.hidden, 
                            learning_rate= self.learning,
                            epochs= self.epochs, 
                            momentum_const=self.momentum, 
                            decay_rate=self.decay_rate,
                            activation=self.activation,
                            dropout=False,
                            minibatch_size=50, 
                            nesterov=True)
                            
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
        return accuracy, train_error, test_error

    def plot_confusion_matrix(self, classifier, names, title='Confusion matrix', cmap=plt.cm.Blues):
        pred = classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, pred)
        fig = plt.figure()
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(names))
        plt.xticks(tick_marks, names, rotation=45)
        plt.yticks(tick_marks, names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig
    
   
