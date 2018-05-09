"""Utils to perform error checking, CV, and hyperparameter tuning."""
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

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
    
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()