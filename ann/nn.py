import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn import preprocessing  
from lib.utils import *
from lib.NeuralNetwork import NeuralNetwork

# CONSTANT VARIABLES
RANDOM_SEED = 45
FILENAME = 'damar.csv'

def encode_text_index(df, name):
    """
        Label Encoding using sklearn.preporcessing. Transforms labels into integers i.e: [a, b, c] => [1, 2, 3]

        df: pandas.DataFrame
        name: string

    """
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

def prepare_data():
    """
        Reads data from file, and splits it into training and testing data

    """
    data =  pd.read_csv(FILENAME, sep=';', decimal=',')
    encode_text_index(data,'sinif')
    classes = data['sinif']
    num_labels = len(np.unique(classes))
    inputs = data.drop('sinif', axis=1)
    train, test, Y, y = train_test_split(inputs,classes,test_size=0.25, random_state = RANDOM_SEED)
    X_train =  train.values
    x_test =  test.values
    Y_train = Y.values
    y_test = y.values
    return X_train, x_test, Y_train, y_test, num_labels


def calculate_accuracy(classifier,X, Y):
    """
        Helper method to find the accuracy of a prediction
        
        classifier: NeuralNetowrk instance
        X: imput data
        Y: target 
    """
    return classifier.accuracy(X, Y)

def main():
    """
        Main Function to iniate the Neural Network training
    """
    start = time.time()
    model_name = "neural.sav"
    X_train, x_test, Y_train, y_test, dimension = prepare_data()
    ann = NeuralNetwork(n_output=dimension, 
                        n_features=X_train.shape[1], 
                        n_hidden=100, 
                        learning_rate=0.001, 
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

    accuracy = calculate_accuracy(ann, X_train, Y_train)
    train_error, test_error = get_train_test_error(ann, X_train, Y_train, num_iterations=1, split= 0.25)
    print ("Training Error: {}, Testing error: {} ".format(train_error, test_error))
    print ("Total accuracy {}".format(accuracy))
    print ("The program exectuted successfuly in: %s seconds" % (time.time() - start))

    
if(__name__ == "__main__"):
    '''
        Main Program'
    '''
    main()