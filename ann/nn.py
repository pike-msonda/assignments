import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from lib.utils import *
from lib.NeuralNetwork import NeuralNetwork

def main():
    """
        Main Function to iniate the Neural Network training
    """
    start = time.time()
    model_name = "neural.sav"
    X_train, x_test, Y_train, y_test, dimension, classes = prepare_data()

    # Initialize Neural Network-> This setup has been tested to give the most optimal results. 
    ann = NeuralNetwork(n_output=dimension, 
                        n_features=X_train.shape[1], 
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
    if(os.path.exists(model_name)):
        print("Model already exists, will now calculate accuracy")
        ann = pickle.load(open(model_name, 'rb'))
    else:
        print("Model not found, we will proceed to train and calculate accuracy")
        ann.fit(X_train, Y_train)
        pickle.dump(ann, open(model_name, 'wb'))

    accuracy = calculate_accuracy(ann, x_test, y_test)
    train_error, test_error = get_train_test_error(ann, x_test, y_test, num_iterations=1, split= 0.33)
    print ("Training Error: {}, Testing error: {} ".format(train_error, test_error))
    print ("Total accuracy {}".format(accuracy))

    #Building a confusion Matrix
    pred = ann.predict(x_test)
    cm = confusion_matrix(y_test, pred)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cm, classes)
    print ("The program exectuted successfuly in: %s seconds" % (time.time() - start))

    
if(__name__ == "__main__"):
    '''
        Main Program'
    '''
    main()
