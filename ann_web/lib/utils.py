import pandas as pd
import numpy as np
import csv
from multiprocessing import Process, Queue
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# CONSTANT VARIABLES
RANDOM_SEED = 250
FILENAME = 'data/damar.csv'
MODELFILENAME = "AI_model/neuron.sav"
IMAGE = ""
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
        
       # classifier.fit(X_train, y_train)
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




def calculate_accuracy(classifier,X, Y):
    """
        Helper method to find the accuracy of a prediction

        classifier: NeuralNetowrk instance
        X: imput data
        Y: target
    """
    return classifier.accuracy(X, Y)

def encode_text_index(df):
    """
        Label Encoding using sklearn.preporcessing. Transforms labels into integers i.e: [a, b, c] => [1, 2, 3]

        df: pandas.DataFrame
        name: string

    """
    le = preprocessing.LabelEncoder()
    df.iloc[:,-1] = le.fit_transform(df.iloc[:,-1])
    return le.classes_

def prepare_data(file):
    """
        Reads data from file, and splits it into training and testing data

    """
    data =  read_data(file)
    classes = encode_text_index(data)
    encoded_classes=data.iloc[:,-1]
    data_values = data.values
    inputs = data_values[:,:-1]
    train, test, Y, y = train_test_split(inputs,encoded_classes,test_size=0.25, random_state = RANDOM_SEED)
    X_train =  train
    x_test =  test
    Y_train = Y
    y_test = y
    return X_train, x_test, Y_train, y_test, classes

def const_values(data):
    classes = data.iloc[:,-1]
    outputs = number_of_labels(classes)
    inputs  = data.shape[1] - 1

    return inputs, outputs

def number_of_labels(classes):
    return len(np.unique(classes))


def read_data(file):
    dialect = csv_dialect(file)
    data =  pd.read_csv(file, dialect=dialect, decimal=',')
    return data


def csv_dialect(file):
    dialect = ""
    with open(file, 'rb') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.readline())
        csvfile.seek(0)
    return dialect

def convert_fig_to_html(fig):
    """ Convert Matplotlib figure 'fig' into a <img> tag for HTML use using base64 encoding. """
    import urllib
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import StringIO
    canvas = FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    return '<img src="data:image/png;base64,{}">'.format(urllib.quote(png_output.getvalue().encode('base64').rstrip('\n')))

def process_start(target, args=[]):
    queue = Queue()
    index = len(args)
    args.insert(index, queue)
    args = tuple(args)
    process = Process(target=target, args=args)
    process.start()
    results = queue.get()
    process.join()

    return results

def trainer(ann, queue):
    model = ann.train()
    queue.put(model)

def graphpainter(ann, model,queue):
    classes = ann.classes
    fig = ann.plot_confusion_matrix(model, classes)
    queue.put(convert_fig_to_html(fig))
