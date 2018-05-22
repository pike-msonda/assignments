import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# CONSTANT VARIABLES
RANDOM_SEED = 200
FILENAME = 'data/damar.csv'
MODELFILENAME = "AI_model/neuron.sav"

def onehot_encode(y):
    """one hot encode labels into a matrix
    Finds the range of labels r and then creates a r x y.shape matrix to onehot encode"""
    min, max = min(y), max(y)
    r = max - min + 1
    Y = np.zeros((r, y.shape[0]))
    for idx, val in enumerate(Y):
        Y[val][idx] = 1
    return Y

def get_train_test_error(classifier, X, y, num_iterations = 1, split = 0.25):
    """Returns the average training and test error over a specified number of
    iterations, for a specified split of the data.
    Params:
        classifier: a classifier with a fit(X, y) and predict(y) API
        X: the training dataset of examples
        y: the testing dataset of examples
        num_iterations: number of iterations to run fit() and predict()
        split: the propoprtion of data that should be reserved for validation.
    """

    train_error, test_error = 0.0, 0.0
    for i in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=split,
                                                            random_state=i)
        #classifier.fit(X_train, y_train)
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        train_error+=1- metrics.accuracy_score(y_true=y_train,
                                               y_pred=y_train_pred,
                                               normalize=True)
        test_error+=1 - metrics.accuracy_score(y_true=y_test,
                                               y_pred=y_test_pred,
                                               normalize=True)
    train_error /=num_iterations
    test_error /=num_iterations
    return train_error, test_error
    
# def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(names))
#     plt.xticks(tick_marks, names, rotation=45)
#     plt.yticks(tick_marks, names)
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()

def calculate_accuracy(classifier,X, Y):
    """
        Helper method to find the accuracy of a prediction
        
        classifier: NeuralNetowrk instance
        X: imput data
        Y: target 
    """
    return classifier.accuracy(X, Y)

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
    data =  read_data()
    encode_text_index(data,'sinif')
    classes = data['sinif']
    inputs = data.drop('sinif', axis=1)
    train, test, Y, y = train_test_split(inputs,classes,test_size=0.25, random_state = RANDOM_SEED)
    X_train =  train.values
    x_test =  test.values
    Y_train = Y.values
    y_test = y.values
    return X_train, x_test, Y_train, y_test, classes

def number_of_labels(classes):
    return len(np.unique(classes))

def read_data():
    data =  pd.read_csv(FILENAME, sep=';', decimal=',')
    return data