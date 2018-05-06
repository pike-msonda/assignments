'''
    Program to train an Artificial Network
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical


def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

#  Transform data to fit the format acceptable by Keras model
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

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

if(__name__ == "__main__"):
    '''
        Main Program'
    '''
    np.random.seed(125)
    # Read CSV file
    import_data = pd.read_csv('damar.csv', delimiter=';', decimal=',')
    #print (import_data.head())
    classes = encode_text_index(import_data, "sinif")
    print (classes)
    x_data, y_data = to_xy(import_data, "sinif")
    print (y_data.shape)
    # Split data set into train and test sets
    X_train_set, X_test_set, Y_train_test, Y_test_set =  train_test_split(x_data, y_data, test_size=0.30)
    print("X {} and Y {} training set count".format(X_train_set.shape, Y_train_test.shape))
    print("X {} and Y {} testing set count ".format(X_test_set.shape, Y_test_set.shape))

    # #create Neural Network 
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=x_data.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.6, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train_set, Y_train_test, verbose=0, epochs=1000, batch_size=32)
    score = model.evaluate(X_test_set, Y_test_set, verbose = 1, batch_size=32)
    print(score)
    pred = model.predict(X_test_set)
    pred = np.argmax(pred, axis=1)
    Y_test2 = np.argmax(Y_test_set, axis=1)
    cm =  confusion_matrix(Y_test2, pred)
    np.set_printoptions(precision=2)
    print ("Confusion Matrix, without Normalisation")
    print (cm)
    plt.figure()
    plot_confusion_matrix(cm, classes)
    # Plot a count graph 
    #ax = sns.countplot(x='sinif', data=import_data, palette=sns.color_palette("Spectral",5))
    plt.show()

